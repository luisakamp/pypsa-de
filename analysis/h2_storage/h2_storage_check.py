# SPDX-FileCopyrightText: : 2026 The PyPSA-Eur Authors
#
# SPDX-License-Identifier: MIT
"""
Phase 0/1 check of hydrogen storage dispatch in solved PyPSA-DE networks.

Self-contained (only needs pypsa + pandas), so it can run e.g. in Databricks:

    %pip install pypsa
    from h2_storage_check import run
    summary, balances = run("/Volumes/.../KN2045_Mix/results", out_dir="/Volumes/.../h2_check")

or from the shell:

    python h2_storage_check.py <results_dir> [--out <dir>] [--years 2030 2035 ...]

Conventions
-----------
- All values in model units: MW / MWh lower heating value (Hi).
  NEP values are in higher heating value: Hs = 1.189 * Hi (columns *_Hs).
- Store dispatch p > 0 means discharge into the network (ASL), p < 0 charging (ESL).
- With multi-hour snapshots, p is the mean power over the snapshot; energies are
  weighted with ``n.snapshot_weightings``.
- Underground caverns and overground tanks share the carrier "H2 Store". A bus counts
  as cavern location if any of its H2 stores has a finite ``e_nom_max``
  (salt cavern potential), otherwise as tank location.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pypsa

HS_PER_HI = 1.189
YEARS = [2025, 2030, 2035, 2040, 2045, 2050]
FILE_PATTERN = "base_s_27__none_{year}.nc"
H2_PIPE_CARRIERS = ["H2 pipeline", "H2 pipeline retrofitted", "H2 pipeline (Kernnetz)"]


def ts(df: pd.DataFrame, idx, index) -> pd.DataFrame:
    """Select columns from a time-series frame; missing (all-zero, not exported) -> 0."""
    return df.reindex(index=index, columns=idx).fillna(0.0)


def de_h2_buses(n: pypsa.Network, country: str = "DE") -> pd.Index:
    b = n.buses
    return b.index[(b.carrier == "H2") & b.index.str.startswith(country)]


def classify_h2_stores(n: pypsa.Network, buses: pd.Index) -> pd.DataFrame:
    st = n.stores[(n.stores.carrier == "H2 Store") & n.stores.bus.isin(buses)].copy()
    cavern_buses = st.bus[np.isfinite(st.e_nom_max)].unique()
    st["type"] = np.where(st.bus.isin(cavern_buses), "cavern", "tank")
    return st


def h2_injections(n: pypsa.Network, buses: pd.Index, country: str = "DE") -> pd.DataFrame:
    """
    Time series of net injection (MW) into the given H2 buses, by carrier.

    Cross-border H2 pipelines are labelled "import" (positive = into DE),
    pipelines inside DE are dropped (they cancel out apart from losses).
    """
    cols = {}
    sns = n.snapshots

    def add(label, series):
        cols[label] = cols.get(label, 0) + series

    # generators (normally none on H2 buses)
    g = n.generators[n.generators.bus.isin(buses)]
    for carrier, idx in g.groupby("carrier").groups.items():
        add(carrier, ts(n.generators_t.p, idx, sns).sum(axis=1))

    # loads (constant p_set loads may not be in loads_t.p if not solved; fall back)
    ld = n.loads[n.loads.bus.isin(buses)]
    for carrier, idx in ld.groupby("carrier").groups.items():
        if set(idx) <= set(n.loads_t.p.columns):
            p = n.loads_t.p[idx]
        else:
            p = n.get_switchable_as_dense("Load", "p_set")[idx]
        add(carrier, -p.sum(axis=1))

    # stores
    st = classify_h2_stores(n, buses)
    for typ, idx in st.groupby("type").groups.items():
        add(f"H2 Store ({typ})", ts(n.stores_t.p, idx, sns).sum(axis=1))
    other_st = n.stores[n.stores.bus.isin(buses) & ~n.stores.index.isin(st.index)]
    for carrier, idx in other_st.groupby("carrier").groups.items():
        add(carrier, ts(n.stores_t.p, idx, sns).sum(axis=1))

    # links, every port
    ports = sorted(int(c[3:]) for c in n.links.columns if c.startswith("bus") and c[3:].isdigit())
    links = n.links
    for i in ports:
        attached = links.index[links[f"bus{i}"].isin(buses)]
        if attached.empty:
            continue
        p_i = ts(n.links_t[f"p{i}"], attached, sns)
        inj = -p_i  # p_i is withdrawal from bus i
        sub = links.loc[attached]
        is_pipe = sub.carrier.isin(H2_PIPE_CARRIERS)
        other = sub["bus1" if i == 0 else "bus0"]
        cross = is_pipe & ~other.str.startswith(country)
        internal = is_pipe & ~cross
        if cross.any():
            add("import (pipelines)", inj[sub.index[cross]].sum(axis=1))
        if internal.any():
            add("DE pipelines (losses)", inj[sub.index[internal]].sum(axis=1))
        rest = sub[~is_pipe]
        for carrier, idx in rest.groupby("carrier").groups.items():
            add(carrier, inj[idx].sum(axis=1))

    df = pd.DataFrame(cols)
    df["balance residual"] = df.sum(axis=1)
    return df


def _div(a, b):
    return a / b if b else np.nan


MIN_GW_FOR_RATIO = 100.0  # MW; below this the ESL/ASL ratio is numerical noise


def duration_stats(p: pd.Series, w: pd.Series, prefix: str) -> dict:
    """ASL/ESL statistics for a store dispatch series in MW (p>0 discharge)."""
    out = {}
    hourly_mean = p  # mean power within snapshot
    daily = (p * w).resample("D").sum() / w.resample("D").sum()
    for lab, s in {"snap": hourly_mean, "day": daily}.items():
        s = s.dropna()
        out[f"{prefix}ASL_max_{lab}"] = s.max()
        out[f"{prefix}ESL_max_{lab}"] = -s.min()
        out[f"{prefix}ASL_q995_{lab}"] = s.quantile(0.995)
        out[f"{prefix}ESL_q995_{lab}"] = -s.quantile(0.005)
    for lab in ["snap", "day"]:
        asl, esl = out[f"{prefix}ASL_max_{lab}"], out[f"{prefix}ESL_max_{lab}"]
        out[f"{prefix}ESL/ASL_max_{lab}"] = (
            _div(esl, asl) if max(asl, esl) >= MIN_GW_FOR_RATIO else np.nan
        )
    dis = (p.clip(lower=0) * w).sum()
    cha = (-p.clip(upper=0) * w).sum()
    out[f"{prefix}discharge_TWh"] = dis / 1e6
    out[f"{prefix}charge_TWh"] = cha / 1e6
    big = max(out[f"{prefix}ASL_max_snap"], out[f"{prefix}ESL_max_snap"]) >= MIN_GW_FOR_RATIO
    out[f"{prefix}FLH_ASL_h"] = _div(dis, out[f"{prefix}ASL_max_snap"]) if big else np.nan
    out[f"{prefix}FLH_ESL_h"] = _div(cha, out[f"{prefix}ESL_max_snap"]) if big else np.nan
    return out


def check_network(n: pypsa.Network, year: int) -> tuple[dict, pd.DataFrame]:
    info: dict = {"year": year}

    # --- time resolution
    sw = n.snapshot_weightings
    info["n_snapshots"] = len(n.snapshots)
    info["weighting_unique_h"] = sorted(sw.generators.unique().round(3).tolist())[:5]
    info["sum_weights_h"] = sw.generators.sum()
    try:
        cfg = n.meta.get("clustering", {}).get("temporal", {})
        info["config_clustering_temporal"] = json.dumps(cfg)
        info["config_snapshots"] = json.dumps(n.meta.get("snapshots", {}))
        info["config_scenario"] = json.dumps(n.meta.get("run", {}).get("name"))
    except AttributeError:
        pass

    buses = de_h2_buses(n)
    info["n_DE_H2_buses"] = len(buses)

    # --- storage
    st = classify_h2_stores(n, buses)
    for typ in ["cavern", "tank"]:
        s = st[st.type == typ]
        info[f"{typ}_buses"] = ",".join(sorted(s.bus.unique()))
        info[f"{typ}_e_nom_opt_TWh"] = s.e_nom_opt.sum() / 1e6
        info[f"{typ}_e_nom_max_TWh"] = (
            s.groupby("bus").e_nom_max.max().replace(np.inf, np.nan).sum() / 1e6
        )
        info[f"{typ}_capital_cost_EUR_per_MWh_a"] = (
            json.dumps(s.capital_cost.round(2).unique().tolist()) if len(s) else ""
        )
    # any power limit on stores? (should be none -> unconstrained ASL/ESL)
    h2_store_links = n.links.index[
        n.links.bus0.isin(buses) & n.links.bus1.str.contains("H2 Store", na=False)
        | n.links.bus1.isin(buses) & n.links.bus0.str.contains("H2 Store", na=False)
    ]
    info["store_charger_links"] = len(h2_store_links)

    # --- balance and dispatch
    inj = h2_injections(n, buses)
    w = sw.generators.reindex(inj.index)
    info["balance_residual_max_abs_MW"] = inj["balance residual"].abs().max()

    store_cols = [c for c in inj.columns if c.startswith("H2 Store")]
    p_store = inj[store_cols].sum(axis=1)
    info.update(duration_stats(p_store, w, "all_"))
    if "H2 Store (cavern)" in inj:
        info.update(duration_stats(inj["H2 Store (cavern)"], w, "cavern_"))

    e = ts(n.stores_t.e, st.index, n.snapshots).sum(axis=1)
    info["AGV_used_TWh"] = (e.max() - e.min()) / 1e6
    info["cycles"] = (
        _div(info["all_discharge_TWh"], info["AGV_used_TWh"]) if info["AGV_used_TWh"] > 0.01 else np.nan
    )

    # --- capacities (DE)
    def cap(carriers, attr="p_nom_opt", by_output=False):
        sel = n.links[n.links.carrier.isin(carriers) & n.links.index.str.startswith("DE")]
        v = sel[attr] * (sel.efficiency if by_output else 1)
        return v.sum() / 1e3

    info["electrolysis_GWel"] = cap(["H2 Electrolysis"])
    h2_power = [c for c in n.links.carrier.unique() if c.startswith("H2") and any(
        k in c for k in ["OCGT", "CCGT", "CHP", "turbine", "Fuel Cell"])]
    h2_power += [c for c in n.links.carrier.unique() if "H2" in c and "CHP" in c]
    h2_power = sorted(set(h2_power))
    info["h2_power_carriers"] = ",".join(h2_power)
    info["h2_power_GWel"] = cap(h2_power, by_output=True)
    info["h2_power_GW_H2in"] = cap(h2_power)

    # --- annual balance (TWh) by carrier
    annual = inj.mul(w, axis=0).sum() / 1e6
    for k, v in annual.items():
        info[f"annual_TWh[{k}]"] = v

    # --- Hs columns for key figures
    for k in ["all_ASL_max_snap", "all_ESL_max_snap", "all_ASL_max_day", "all_ESL_max_day"]:
        info[k + "_GW_Hs"] = info[k] / 1e3 * HS_PER_HI

    return info, inj


def run(results_dir, years=YEARS, out_dir=None, pattern=FILE_PATTERN):
    results_dir = Path(results_dir)
    rows, balances = [], {}
    for y in years:
        f = results_dir / pattern.format(year=y)
        if not f.exists():
            print(f"missing: {f}")
            continue
        n = pypsa.Network(f)
        info, inj = check_network(n, y)
        rows.append(info)
        balances[y] = inj
        print(
            f"{y}: {info['n_snapshots']} snapshots | ASL {info['all_ASL_max_snap'] / 1e3:.1f} GW, "
            f"ESL {info['all_ESL_max_snap'] / 1e3:.1f} GW (Hi), ESL/ASL {info['all_ESL/ASL_max_snap']:.0%} | "
            f"AGV {info['AGV_used_TWh']:.1f} TWh | residual {info['balance_residual_max_abs_MW']:.1f} MW"
        )
    summary = pd.DataFrame(rows).set_index("year").T
    if out_dir:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        summary.to_csv(out / "h2_storage_summary.csv")
        for y, inj in balances.items():
            inj.to_csv(out / f"h2_balance_DE_{y}.csv")
    return summary, balances


def _in_notebook() -> bool:
    try:
        from IPython import get_ipython

        return get_ipython() is not None
    except ImportError:
        return False


if __name__ == "__main__" and not _in_notebook():
    # In a notebook (Databricks/Jupyter) the cell only defines the functions;
    # call run(...) afterwards instead of using the command line interface.
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir")
    ap.add_argument("--out", default=None)
    ap.add_argument("--years", type=int, nargs="*", default=YEARS)
    ap.add_argument("--pattern", default=FILE_PATTERN)
    a = ap.parse_args()
    s, _ = run(a.results_dir, a.years, a.out, a.pattern)
    with pd.option_context("display.max_rows", 200, "display.width", 200):
        print(s)
