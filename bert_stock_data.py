"""Loader / mapping for the BERT engine's stock data bundle.

build_stock_bundle.py writes stock_data_bundle.npz: the
genetic_lab price panel (data/prices_pit) and volume sidecar
(data/volume_sidecar) aligned on the SPY trading calendar,
plus optional point-in-time fundamental series derived from
the EDGAR companyfacts store. This module is the mapping
from that file to the panel contract bert_domain_stock and
bert_preprocessor consume:

    .open / .high / .low / .close / .volume  float32 (A, T)
    .prices   alias of .close (the domain's marking prices)
    .tickers  [str], .dates [str] (YYYY-MM-DD, increasing)
    .fundamentals  {name: float32 (A, T)} ({} when the bundle
                   carries no fund_<name> arrays)
    .train_end_index  int: number of calendar days with
                   year <= 2019 (the GA training window); the
                   stock domain uses it as the default
                   preprocessor stats_end for bundle panels.

Frozen validation episodes (validation_episodes_v1.json)
carry basket_panel_indices into the 622-ticker genetic_lab
panel ordering of 2026-11-15, NOT this bundle's ordering.
remap_episodes() resolves baskets by ticker NAME into bundle
rows (and validates day ranges); check_episodes() reports
the same without raising.
"""

from __future__ import annotations

import json

import numpy as np

_ARRAYS = ("open", "high", "low", "close", "volume")


class BundlePanel:
    """In-memory panel over a stock data bundle. Arrays are
    float32 (A, T); missing bars are NaN (never filled)."""

    def __init__(self, open_, high, low, close, volume,
                 tickers, dates, fundamentals,
                 train_end_index, meta=None):
        self.open = open_
        self.high = high
        self.low = low
        self.close = close
        self.volume = volume
        self.prices = close
        self.tickers = list(tickers)
        self.dates = list(dates)
        self.fundamentals = dict(fundamentals)
        self.train_end_index = int(train_end_index)
        self.meta = dict(meta or {})

    @property
    def shape(self):
        return self.close.shape


def load_bundle_panel(path):
    """Load stock_data_bundle.npz -> BundlePanel.

    Validates: every OHLCV array has the same (A, T) shape
    with A == len(tickers) and T == len(dates); tickers are
    unique; dates are strictly increasing; fundamental arrays
    match (A, T). Raises ValueError with a specific message
    on any violation.
    """
    z = np.load(path, allow_pickle=False)
    keys = set(z.files)
    missing = [k for k in (*_ARRAYS, "tickers", "dates",
                           "train_end_index") if k not in keys]
    if missing:
        raise ValueError(
            f"bundle {path} is missing arrays: {missing}")
    tickers = [str(t) for t in z["tickers"]]
    dates = [str(d) for d in z["dates"]]
    if len(set(tickers)) != len(tickers):
        raise ValueError(
            f"bundle {path}: tickers are not unique")
    if any(dates[i] >= dates[i + 1]
           for i in range(len(dates) - 1)):
        raise ValueError(
            f"bundle {path}: dates are not strictly increasing")
    A, T = len(tickers), len(dates)
    arrays = {}
    for name in _ARRAYS:
        a = np.asarray(z[name], dtype=np.float32)
        if a.shape != (A, T):
            raise ValueError(
                f"bundle {path}: array {name!r} has shape "
                f"{a.shape}, expected {(A, T)}")
        arrays[name] = a
    fundamentals = {}
    for k in sorted(keys):
        if k.startswith("fund_"):
            a = np.asarray(z[k], dtype=np.float32)
            if a.shape != (A, T):
                raise ValueError(
                    f"bundle {path}: fundamental array {k!r} "
                    f"has shape {a.shape}, expected {(A, T)}")
            fundamentals[k[len("fund_"):]] = a
    meta = {}
    if "meta" in keys:
        meta = json.loads(str(z["meta"]))
    return BundlePanel(arrays["open"], arrays["high"],
                       arrays["low"], arrays["close"],
                       arrays["volume"], tickers, dates,
                       fundamentals,
                       int(z["train_end_index"]), meta)


def check_episodes(episodes, panel, episode_days=60):
    """Non-raising audit of frozen episodes against a panel:
    every episode ticker must exist in panel.tickers and
    start + episode_days must fit the calendar. Returns a
    report dict {n_episodes, ok, missing_tickers (sorted),
    out_of_range (episode indices), max_start}."""
    have = set(panel.tickers)
    T = len(panel.dates)
    missing = set()
    out_of_range = []
    max_start = -1
    for i, ep in enumerate(episodes):
        for t in ep["tickers"]:
            if t not in have:
                missing.add(t)
        max_start = max(max_start, int(ep["start"]))
        if int(ep["start"]) < 0 \
                or int(ep["start"]) + int(episode_days) > T:
            out_of_range.append(i)
    return {"n_episodes": len(episodes),
            "ok": not missing and not out_of_range,
            "missing_tickers": sorted(missing),
            "out_of_range": out_of_range,
            "max_start": max_start,
            "n_days": T}


def remap_episodes(episodes, panel, episode_days=60):
    """Resolve episode baskets by ticker NAME into panel row
    indices (the frozen baskets' stored indices address a
    different panel ordering). Returns new episode dicts with
    'basket' as an int array of panel rows, in the episodes'
    own ticker order. Raises ValueError listing every problem
    when a ticker is absent or a window leaves the calendar.
    """
    rep = check_episodes(episodes, panel, episode_days)
    if not rep["ok"]:
        raise ValueError(
            f"episodes do not resolve against this panel: "
            f"missing tickers {rep['missing_tickers']}, "
            f"out-of-range episodes {rep['out_of_range']}")
    row = {t: i for i, t in enumerate(panel.tickers)}
    out = []
    for ep in episodes:
        out.append({
            "basket": np.array([row[t] for t in ep["tickers"]],
                               dtype=int),
            "tickers": list(ep["tickers"]),
            "start": int(ep["start"])})
    return out
