"""Build stock_data_bundle.npz for the BERT engine's stock
domain: the genetic_lab price panel + volume sidecar, aligned
on the SPY trading calendar, plus point-in-time fundamentals
derived from the EDGAR companyfacts store.

Sources (defaults under ~/workspace/trading-agent):
  data/prices_pit/*.csv      date,open,high,low,close per
                             ticker (Yahoo symbols as stems)
  data/volume_sidecar/*.csv  date,volume per ticker (same
                             harvest; ^VIX has no file)
  data/fundamentals/         EDGAR companyfacts cache + cik
                             map (the fundamental strategies'
                             store)

Alignment contract (mirrors genetic_lab/alloc_v3_env.Panel):
the calendar is SPY.csv's date list; every ticker is
reindexed onto it; days a ticker has no bar stay NaN (never
filled); rows dated outside the calendar are dropped and
counted. The builder asserts per ticker that every volume
date exists in that ticker's price dates and REPORTS any
violation (it does not silently pass, and it does not abort
the build for it).

Fundamentals (fund_<name> arrays, float32 (A, T)) are derived
with the trading program's own frozen machinery, not new
definitions: trading_agent.strategies_fund._FactStore (the
strategies' point-in-time store: filed <= as_of, first-vintage
per period) for values, strategies_fund.TAG_TOTAL_LIABILITIES
and fundamentals.TAG_* frozen tag lists for concepts, and
trading_agent.splits.adjusted_shares_asof for the
audit-corrected (2026-11-10) split-adjusted share count, so
market caps match the corrected convention the strategies
were re-run on. Each component is a step function of its own
filings: known from its filed date, forward-filled, NaN
before first availability.

  earnings_yield   = TTM net income / corrected market cap
  book_to_market   = book equity / corrected market cap
  return_on_assets = TTM net income / total assets
  debt_to_assets   = total liabilities / total assets
(corrected market cap = split-adjusted shares x close of the
day; TTM = _FactStore.ttm over TAG_NET_INCOME; balance-sheet
items = _FactStore.instant_asof. Ratios are NaN wherever an
input is missing or a denominator is not positive.)

Usage:
  python3 build_stock_bundle.py [--out PATH] [--prices-dir D]
      [--volume-dir D] [--fundamentals-dir D] [--ta-root D]
      [--no-fundamentals]
Prints a build report (size, coverage, NaN shares, stale
tickers, assertion violations, fundamentals coverage).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
TA_ROOT = HERE.parent.parent / "trading-agent"
PRICE_FIELDS = ("open", "high", "low", "close")
FUND_NAMES = ("earnings_yield", "book_to_market",
              "return_on_assets", "debt_to_assets")


# ------------------------------------------------------------------ prices
def read_calendar(prices_dir: Path) -> list[str]:
    df = pd.read_csv(prices_dir / "SPY.csv", dtype={"date": str})
    dates = [str(d) for d in df["date"]]
    if dates != sorted(dates) or len(set(dates)) != len(dates):
        raise ValueError("SPY calendar is not strictly "
                         "increasing / unique")
    return dates


def build_price_arrays(prices_dir: Path, volume_dir: Path,
                       tickers: list[str], dates: list[str]):
    """Reindex every ticker onto the SPY calendar. Returns
    (arrays {name: float32 (A, T)}, report dict)."""
    A, T = len(tickers), len(dates)
    cal_index = {d: i for i, d in enumerate(dates)}
    arrays = {f: np.full((A, T), np.nan, dtype=np.float32)
              for f in (*PRICE_FIELDS, "volume")}
    report = {"duplicate_dates": [], "noncalendar_rows": 0,
              "stale_price": [], "stale_volume": [],
              "no_volume_file": [],
              "volume_date_violations": []}
    last_day = dates[-1]
    for a, tk in enumerate(tickers):
        df = pd.read_csv(prices_dir / f"{tk}.csv",
                         dtype={"date": str}).sort_values("date")
        dstr = [str(d) for d in df["date"]]
        if len(set(dstr)) != len(dstr):
            report["duplicate_dates"].append(tk)
        price_dates = set(dstr)
        idx = np.array([cal_index.get(d, -1) for d in dstr])
        ok = idx >= 0
        report["noncalendar_rows"] += int((~ok).sum())
        for f in PRICE_FIELDS:
            arrays[f][a, idx[ok]] = df[f].to_numpy(
                dtype=np.float32)[ok]
        if dstr and dstr[-1] < last_day:
            report["stale_price"].append(
                {"ticker": tk, "last_date": dstr[-1]})
        vf = volume_dir / f"{tk}.csv"
        if not vf.exists():
            report["no_volume_file"].append(tk)
            continue
        dv = pd.read_csv(vf, dtype={"date": str}) \
            .sort_values("date")
        vstr = [str(d) for d in dv["date"]]
        bad = [d for d in vstr if d not in price_dates]
        if bad:
            report["volume_date_violations"].append(
                {"ticker": tk, "n_bad": len(bad),
                 "first_bad": bad[0]})
        vidx = np.array([cal_index.get(d, -1) for d in vstr])
        vok = vidx >= 0
        arrays["volume"][a, vidx[vok]] = dv["volume"].to_numpy(
            dtype=np.float32)[vok]
        if vstr and vstr[-1] < last_day:
            report["stale_volume"].append(
                {"ticker": tk, "last_date": vstr[-1]})
    return arrays, report


# ------------------------------------------------------------------ fundamentals
def _step_on_calendar(event_dates, values, dates):
    """Step function: value known from its event (filed) date,
    forward-filled over the calendar; NaN before the first
    event. Later events at the same calendar position win
    (they carry the larger ordinal)."""
    T = len(dates)
    vals = np.array([np.nan if v is None else float(v)
                     for v in values])
    pos = np.searchsorted(np.asarray(dates), np.asarray(
        event_dates), side="left")
    ordinal = np.full(T, -1, dtype=int)
    for k, p in enumerate(pos):
        if p < T:
            ordinal[p] = k
    ordinal = np.maximum.accumulate(ordinal)
    out = np.full(T, np.nan)
    known = ordinal >= 0
    out[known] = vals[ordinal[known]]
    return out


def build_fundamentals(tickers, dates, close, ta_root: Path,
                       fund_dir: Path):
    """Derive the four as-of fundamental series per ticker
    with the trading program's own store. Returns
    ({name: float32 (A, T)}, report dict)."""
    sys.path.insert(0, str(ta_root))
    import trading_agent.strategies_fund as sf
    from trading_agent import fundamentals as F
    from trading_agent.splits import (adjusted_shares_asof,
                                      load_splits)

    store = sf._get_store({"cache_dir": str(fund_dir)})
    splits = load_splits()
    # bundle tickers are Yahoo symbols (file stems); the fact
    # store is keyed by membership ticker. The coverage file
    # is the program's own symbol map.
    yahoo2member = {}
    cov = ta_root / "data" / "pit_yahoo_coverage.csv"
    if cov.exists():
        cdf = pd.read_csv(cov, dtype=str)
        for _, r in cdf.iterrows():
            if isinstance(r.get("yahoo_symbol"), str):
                yahoo2member.setdefault(r["yahoo_symbol"],
                                        r["ticker"])
    tag_lists = {"ni": F.TAG_NET_INCOME,
                 "assets": F.TAG_TOTAL_ASSETS,
                 "equity": F.TAG_BOOK_EQUITY,
                 "liab": sf.TAG_TOTAL_LIABILITIES,
                 "shares": F.TAG_SHARES_OUTSTANDING}
    A, T = len(tickers), len(dates)
    comp = {k: np.full((A, T), np.nan)
            for k in ("ni", "assets", "equity", "liab",
                      "shares")}
    report = {"tickers_with_facts": 0, "tickers_no_facts": []}
    for a, tk in enumerate(tickers):
        member = yahoo2member.get(tk, tk)
        if not store.has_facts(member):
            report["tickers_no_facts"].append(tk)
            continue
        report["tickers_with_facts"] += 1
        raw = store._load(member)

        def _events(tags):
            seen = set()
            for tag in tags:
                for row in raw.get(tag, []):
                    seen.add(str(row["filed"])[:10])
            return sorted(seen)

        def _instant(tags, d):
            hit = store.instant_asof(member, tags, d)
            return None if hit is None else float(hit[0])

        grids = {
            "ni": (_events(tag_lists["ni"]),
                   lambda d: store.ttm(member, tag_lists["ni"],
                                       d)),
            "assets": (_events(tag_lists["assets"]),
                       lambda d: _instant(tag_lists["assets"],
                                          d)),
            "equity": (_events(tag_lists["equity"]),
                       lambda d: _instant(tag_lists["equity"],
                                          d)),
            "liab": (_events(tag_lists["liab"]),
                     lambda d: _instant(tag_lists["liab"], d)),
            "shares": (_events(tag_lists["shares"]),
                       lambda d: (adjusted_shares_asof(
                           splits, store, member, tk, d)
                           or (None,))[0]),
        }
        for name, (ev_dates, fn) in grids.items():
            if not ev_dates:
                continue
            vals = [fn(d) for d in ev_dates]
            comp[name][a] = _step_on_calendar(ev_dates, vals,
                                              dates)
    close64 = close.astype(np.float64)
    mktcap = comp["shares"] * close64
    with np.errstate(divide="ignore", invalid="ignore"):
        ey = np.where((mktcap > 0) & np.isfinite(comp["ni"]),
                      comp["ni"] / np.where(mktcap > 0, mktcap,
                                            np.nan), np.nan)
        btm = np.where((mktcap > 0)
                       & np.isfinite(comp["equity"]),
                       comp["equity"] / np.where(mktcap > 0,
                                                 mktcap, np.nan),
                       np.nan)
        roa = np.where((comp["assets"] > 0)
                       & np.isfinite(comp["ni"]),
                       comp["ni"] / np.where(comp["assets"] > 0,
                                             comp["assets"],
                                             np.nan), np.nan)
        dta = np.where((comp["assets"] > 0)
                       & np.isfinite(comp["liab"]),
                       comp["liab"] / np.where(comp["assets"] > 0,
                                               comp["assets"],
                                               np.nan), np.nan)
    out = {"earnings_yield": ey, "book_to_market": btm,
           "return_on_assets": roa, "debt_to_assets": dta}
    return ({k: v.astype(np.float32) for k, v in out.items()},
            report)


# ------------------------------------------------------------------ bundle
def build_bundle(prices_dir, volume_dir, out_path,
                 fundamentals_dir=None, ta_root=TA_ROOT,
                 include_fundamentals=True):
    """Build the bundle npz and return the build report."""
    prices_dir, volume_dir = Path(prices_dir), Path(volume_dir)
    out_path = Path(out_path)
    tickers = sorted(p.stem for p in prices_dir.glob("*.csv"))
    dates = read_calendar(prices_dir)
    arrays, report = build_price_arrays(prices_dir, volume_dir,
                                        tickers, dates)
    train_end = int(sum(1 for d in dates if int(d[:4]) <= 2019))
    payload = {f: arrays[f] for f in (*PRICE_FIELDS, "volume")}
    payload["tickers"] = np.array(tickers)
    payload["dates"] = np.array(dates)
    payload["train_end_index"] = np.array(train_end)
    fund_report = None
    if include_fundamentals:
        funds, fund_report = build_fundamentals(
            tickers, dates, arrays["close"], Path(ta_root),
            Path(fundamentals_dir))
        for name, arr in funds.items():
            payload[f"fund_{name}"] = arr
    meta = {
        "source": "genetic_lab panel: data/prices_pit (OHLC) + "
                  "data/volume_sidecar (volume), SPY calendar; "
                  "fundamentals derived from the EDGAR "
                  "companyfacts store by build_stock_bundle.py "
                  "with the trading program's frozen tag lists "
                  "and point-in-time store",
        "built_from": {"prices_dir": str(prices_dir),
                       "volume_dir": str(volume_dir),
                       "fundamentals_dir":
                           str(fundamentals_dir)},
        "n_tickers": len(tickers), "n_days": len(dates),
        "first_date": dates[0], "last_date": dates[-1],
        "train_end_index": train_end,
        "fundamentals": sorted(k[len("fund_"):] for k in payload
                               if k.startswith("fund_")),
        "price_report": report,
        "fundamentals_report": fund_report,
    }
    payload["meta"] = np.array(json.dumps(meta))
    np.savez_compressed(out_path, **payload)
    report_out = {
        "out": str(out_path),
        "size_bytes": out_path.stat().st_size,
        "n_tickers": len(tickers), "n_days": len(dates),
        "first_date": dates[0], "last_date": dates[-1],
        "train_end_index": train_end,
        "nan_share": {f: float(np.isnan(arrays[f]).mean())
                      for f in (*PRICE_FIELDS, "volume")},
        **report,
        "fundamentals": fund_report,
        "fundamentals_included": sorted(
            k[len("fund_"):] for k in payload
            if k.startswith("fund_")),
    }
    if fund_report is not None:
        fund_nan = {}
        for name in FUND_NAMES:
            arr = payload.get(f"fund_{name}")
            fund_nan[name] = (None if arr is None else
                              float(np.isnan(arr).mean()))
        report_out["fundamentals_nan_share"] = fund_nan
    return report_out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE /
                                         "stock_data_bundle.npz"))
    ap.add_argument("--prices-dir",
                    default=str(TA_ROOT / "data" / "prices_pit"))
    ap.add_argument("--volume-dir",
                    default=str(TA_ROOT / "data" /
                                "volume_sidecar"))
    ap.add_argument("--fundamentals-dir",
                    default=str(TA_ROOT / "data" /
                                "fundamentals"))
    ap.add_argument("--ta-root", default=str(TA_ROOT))
    ap.add_argument("--no-fundamentals", action="store_true")
    args = ap.parse_args(argv)
    t0 = time.time()
    rep = build_bundle(args.prices_dir, args.volume_dir,
                       args.out, args.fundamentals_dir,
                       Path(args.ta_root),
                       include_fundamentals=not
                       args.no_fundamentals)
    print(json.dumps(rep, indent=1, default=str))
    print(f"[bundle] built {rep['out']}: "
          f"{rep['size_bytes'] / 1e6:.1f} MB, "
          f"{rep['n_tickers']} tickers x {rep['n_days']} days, "
          f"in {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
