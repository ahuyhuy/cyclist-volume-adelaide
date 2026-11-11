#!/usr/bin/env python3
"""
MODEL_REFRESH_TOOL.py

Refresh the Greater Adelaide cyclist volume model from new count data.

Usage (from the repository root):
    python MODEL_REFRESH_TOOL.py

Inputs (relative to the repository root):
    count_data/counts.csv
    graph_generation/graph_cache/graph_loaded.graphml

Outputs (written to a new timestamped folder, model_refreshed_YY_MM_DD_HH_MM/):
    segments/<CBD|CBDP|S|SP>/  X_y.csv, correlation_matrix.csv, regression_summary.txt
    regression_summary.txt          combined model
    model_YY_MM_DD_HH_MM.gpkg       predicted edges (layer: edges_pred)
    model_preview.png               quick-look map
    run_settings.txt                settings and inputs used for this run

Each step mirrors a code cell in model_docs/model_notebook.ipynb.
"""

from __future__ import annotations

import sys
import time
import datetime as dt
import traceback
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration (edit here, not in the functions below)
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent

COUNTS_PATH = REPO_ROOT / "count_data" / "counts.csv"
GRAPH_PATH = REPO_ROOT / "graph_generation" / "graph_cache" / "graph_loaded.graphml"

COUNT_REQUIRED_COLS = ["total-count", "x", "y"]

# Count points further than this from any edge are dropped (metres).
MAX_SNAP_M = 25.0

# OSM relation for the CBD boundary (Central Adelaide LGA, incl. North Adelaide).
BOUNDARY_OSM_ID = "R3093825"

# Segments with fewer count points than this trigger a warning / pause.
MIN_ROWS_PER_SEGMENT = 10

# Predictor lists per segment (must match model_docs/model_notebook.ipynb).
CBD_PREDICTORS = ["POW_CTW", "POW_FE", "PD_MB", "DNL_DZN", "M_30_40",
                  "F_15_25", "INF_CR", "MBI_PTL", "INF_LgRd", "INF_SmRd"]
CBDP_PREDICTORS = ["POW_CTW", "POW_FE", "DNL_DZN", "INF_CR", "MBI_BP",
                   "MBI_PTL", "MBI_SZ", "MBI_Bus", "INF_SmRd"]
S_PREDICTORS = ["POW_CTW", "POW_FE", "PD_MB", "DNL_DZN", "M_30_40",
                "F_15_25", "H_3500", "INF_CR", "INF_TrRd",
                "INF_LgRd", "INF_SmRd"]
SP_PREDICTORS = ["POW_CTW", "POW_FE", "DNL_DZN", "INF_CR", "MBI_PP",
                 "INF_TrRd", "INF_LgRd", "MBI_BP"]

SEGMENT_PREDICTORS = {
    "CBD": CBD_PREDICTORS,
    "CBDP": CBDP_PREDICTORS,
    "S": S_PREDICTORS,
    "SP": SP_PREDICTORS,
}
SEGMENTS = list(SEGMENT_PREDICTORS)

# Edge attributes copied from the graph onto count points during the snap.
EDGE_ATTR_COLS = [
    "INF_CR", "MBI_BP", "MBI_PP", "MBI_PTL", "MBI_Sh", "MBI_SZ", "MBI_Bus", "MBI_Mx",
    "DNL_DZN", "PD_MB", "POW_CTW", "POW_FE", "M_30_40", "F_15_25", "H_3500",
    "INF_TrRd", "INF_LgRd", "INF_SmRd",
]

GRAPH_RERUN_MSG = (
    "Download graph from link in README.md or Run GRAPH_SETUP.ipynb then GRAPH_LOAD.ipynb to create it."
)
GRAPH_MISSING_MSG = (
    "Graph file not found at graph_generation/graph_cache/graph_loaded.graphml.\n"
    f"  {GRAPH_RERUN_MSG}"
)

# import name -> pip name, used for the preflight check.
REQUIRED_PACKAGES = {
    "osmnx": "osmnx",
    "geopandas": "geopandas",
    "networkx": "networkx",
    "numpy": "numpy",
    "pandas": "pandas",
    "scipy": "scipy",
    "statsmodels": "statsmodels",
    "shapely": "shapely",
    "matplotlib": "matplotlib",
    "tqdm": "tqdm",
}


# ---------------------------------------------------------------------------
# Errors and reporting
# ---------------------------------------------------------------------------

class RefreshError(Exception):
    """A user-facing error. The message is shown as-is and the run stops."""


WARNINGS: list[str] = []


def stage(title: str) -> None:
    print(f"\n==> {title}", flush=True)


def info(msg: str) -> None:
    print(f"    {msg}", flush=True)


def warn(msg: str) -> None:
    WARNINGS.append(msg)
    print(f"    WARNING: {msg}", flush=True)


def interactive() -> bool:
    return sys.stdin is not None and sys.stdin.isatty()


def pause_for_weak_segment(segment: str, n: int) -> None:
    """Warn about a weak segment and wait for Enter (only in an interactive terminal)."""
    warn(f"Segment {segment} has only {n} count points (minimum {MIN_ROWS_PER_SEGMENT}). "
         "Results for this segment will be unreliable.")
    if interactive():
        try:
            input("    Press Enter to proceed, or Ctrl+C to stop... ")
        except (KeyboardInterrupt, EOFError):
            raise RefreshError("Stopped by user at the weak-segment warning.")
    else:
        info("Non-interactive run: continuing with the warning recorded in run_settings.txt.")


# ---------------------------------------------------------------------------
# Step 0. Preflight
# ---------------------------------------------------------------------------

def check_packages() -> None:
    missing = []
    for import_name, pip_name in REQUIRED_PACKAGES.items():
        try:
            __import__(import_name)
        except ImportError:
            missing.append(pip_name)
    if missing:
        raise RefreshError(
            f"Missing package(s): {', '.join(missing)}.\n"
            "  Install the dependencies from requirements_conda.txt and requirements_pip.txt, then run again."
        )


def check_inputs() -> None:
    if not COUNTS_PATH.exists():
        raise RefreshError(
            f"Count data not found at count_data/counts.csv (looked in {COUNTS_PATH}).\n"
            "  Place the file there, relative to the repo root, then run again."
        )
    if not GRAPH_PATH.exists():
        raise RefreshError(GRAPH_MISSING_MSG)


def make_output_dir(stamp: str) -> Path:
    out = REPO_ROOT / f"model_refreshed_{stamp}"
    n = 2
    while out.exists():
        out = REPO_ROOT / f"model_refreshed_{stamp}_{n}"
        n += 1
    out.mkdir(parents=True)
    return out


def preflight(stamp: str) -> Path:
    stage("Step 0: Preflight checks")
    check_packages()
    info("Packages OK")
    check_inputs()
    info(f"Count data: {COUNTS_PATH.relative_to(REPO_ROOT)}")
    info(f"Graph:      {GRAPH_PATH.relative_to(REPO_ROOT)} ({GRAPH_PATH.stat().st_size / 1e9:.2f} GB)")
    out = make_output_dir(stamp)
    info(f"Output folder: {out.relative_to(REPO_ROOT)}")
    return out


# ---------------------------------------------------------------------------
# Step 1. Load the graph
# ---------------------------------------------------------------------------

def load_graph():
    import threading
    import osmnx as ox
    from tqdm import tqdm

    stage("Step 1: Loading graph (est. time 2:30)")

    # GraphML gives no progress, so the load runs in a worker thread while the main
    # thread refreshes an elapsed-time display every second.
    result: dict = {}

    def _load():
        try:
            result["G"] = ox.load_graphml(filepath=str(GRAPH_PATH))
        except Exception as exc:  # re-raised on the main thread below
            result["error"] = exc

    worker = threading.Thread(target=_load, daemon=True)
    worker.start()

    with tqdm(total=None, desc="    reading GraphML", bar_format="{desc}: {elapsed}",
              mininterval=1.0) as bar:
        while worker.is_alive():
            bar.refresh()
            worker.join(timeout=1.0)
        bar.refresh()

    if "error" in result:
        raise RefreshError(
            f"Graph file is unreadable or incomplete ({result['error']}).\n  {GRAPH_RERUN_MSG}"
        ) from result["error"]

    G = result["G"]
    info(f"Loaded graph: {len(G.nodes)} nodes, {len(G.edges)} edges")
    return G


# ---------------------------------------------------------------------------
# Step 2. Load and clean count data
# ---------------------------------------------------------------------------

def load_counts():
    import pandas as pd

    stage("Step 2: Loading count data")
    try:
        raw = pd.read_csv(COUNTS_PATH)
    except Exception as exc:
        raise RefreshError(f"Could not read count_data/counts.csv: {exc}") from exc

    missing = [c for c in COUNT_REQUIRED_COLS if c not in raw.columns]
    if missing:
        raise RefreshError(
            f"Missing column(s) {missing} in counts.csv.\n"
            f"  Required: {COUNT_REQUIRED_COLS}. Found: {list(raw.columns)}."
        )

    table = raw[COUNT_REQUIRED_COLS].copy().rename(columns={"total-count": "COUNT"})
    for col in ["COUNT", "x", "y"]:
        table[col] = pd.to_numeric(table[col], errors="coerce")
    table = table.dropna(subset=["COUNT", "x", "y"]).reset_index(drop=True)

    if len(table) == 0:
        raise RefreshError(
            "No usable rows after cleaning. Check counts.csv has numeric x, y and total-count values."
        )
    info(f"Rows loaded: {len(table)} (from {len(raw)} in file)")
    return table


# ---------------------------------------------------------------------------
# Step 3. Snap to network and trim
# ---------------------------------------------------------------------------

BOUNDARY_CACHE_PATH = REPO_ROOT / "graph_generation" / "graph_cache" / f"boundary_{BOUNDARY_OSM_ID}.gpkg"
_BOUNDARY_POLY = None  # in-memory cache for this run


def fetch_boundary():
    """Return the CBD boundary polygon (EPSG:4326). Cached in memory and on disk.

    The first call geocodes the OSM relation (needs internet) and saves it to
    BOUNDARY_CACHE_PATH; later calls, in this run or the next, reuse it.
    """
    global _BOUNDARY_POLY
    if _BOUNDARY_POLY is not None:
        return _BOUNDARY_POLY

    import geopandas as gpd

    if BOUNDARY_CACHE_PATH.exists():
        try:
            _BOUNDARY_POLY = gpd.read_file(BOUNDARY_CACHE_PATH).geometry.iloc[0]
            info(f"Using cached boundary {BOUNDARY_OSM_ID} ({BOUNDARY_CACHE_PATH.name})")
            return _BOUNDARY_POLY
        except Exception as exc:
            warn(f"Cached boundary unreadable ({exc}); fetching again.")

    import osmnx as ox
    import osmnx.settings as ox_settings

    # Compatibility patch, same as the notebook (osmnx 2.x settings names).
    for name, default in [
        ("http_user_agent", "default_user_agent"),
        ("http_referer", "default_referer"),
        ("nominatim_url", "nominatim_endpoint"),
        ("http_accept_language", "default_accept_language"),
    ]:
        if not hasattr(ox_settings, name) and hasattr(ox_settings, default):
            setattr(ox_settings, name, getattr(ox_settings, default))

    try:
        boundary = ox.geocode_to_gdf(BOUNDARY_OSM_ID, by_osmid=True).to_crs("EPSG:4326")
    except Exception as exc:
        raise RefreshError(
            f"Could not fetch the OSM boundary {BOUNDARY_OSM_ID} ({exc}).\n"
            "  Check your internet connection and try again."
        ) from exc

    try:
        boundary.to_file(BOUNDARY_CACHE_PATH, driver="GPKG")
        info(f"Cached boundary {BOUNDARY_OSM_ID} to {BOUNDARY_CACHE_PATH.name}")
    except Exception as exc:  # caching is an optimisation; never block the run on it
        warn(f"Could not cache boundary ({exc}); it will be fetched again next run.")

    _BOUNDARY_POLY = boundary.geometry.iloc[0]
    return _BOUNDARY_POLY


def snap_and_trim(table, G):
    import geopandas as gpd
    import osmnx as ox
    import pandas as pd

    stage("Step 3: Snapping counts to nearest edge and trimming")
    _nodes, edges_gdf = ox.graph_to_gdfs(G, nodes=True, edges=True)
    edges_gdf = edges_gdf.reset_index()
    edges_gdf["_u_v"] = [tuple(sorted((u, v))) for u, v in zip(edges_gdf["u"], edges_gdf["v"])]
    edges_phys = edges_gdf.drop_duplicates(subset=["_u_v", "key"]).reset_index(drop=True)

    metric_crs = edges_phys.estimate_utm_crs()
    edges_metric = edges_phys[["u", "v", "key", "geometry"]].to_crs(metric_crs)

    count_points = gpd.GeoDataFrame(
        table[["COUNT", "x", "y"]].copy(),
        geometry=gpd.points_from_xy(table["x"], table["y"]),
        crs="EPSG:4326",
    ).to_crs(metric_crs)

    snapped = gpd.sjoin_nearest(
        count_points[["geometry"]], edges_metric, how="left", distance_col="dist_to_edge_m",
    )
    snapped = snapped[~snapped.index.duplicated(keep="first")].sort_index()

    table = table.copy()
    table["dist_to_edge_m"] = snapped["dist_to_edge_m"].values
    table["u"] = snapped["u"].values
    table["v"] = snapped["v"].values
    table["key"] = snapped["key"].values

    edge_attrs = edges_phys.set_index(["u", "v", "key"])
    attr_index = pd.MultiIndex.from_arrays([table["u"], table["v"], table["key"]])
    for col in EDGE_ATTR_COLS:
        if col in edge_attrs.columns:
            table[col] = edge_attrs[col].reindex(attr_index).values
        else:
            warn(f"Edge attribute {col} not found on graph; filled with 0. "
                 "Re-run GRAPH_SETUP.ipynb then GRAPH_LOAD.ipynb if this is unexpected.")
            table[col] = 0

    info(f"Snap distance (m): min={table['dist_to_edge_m'].min():.1f}, "
         f"median={table['dist_to_edge_m'].median():.1f}, max={table['dist_to_edge_m'].max():.1f}")

    too_far = table["dist_to_edge_m"] > MAX_SNAP_M
    if too_far.any():
        info(f"Dropping {int(too_far.sum())} of {len(table)} points further than {MAX_SNAP_M} m from an edge:")
        for _, r in table.loc[too_far].iterrows():
            info(f"  x={r['x']:.5f} y={r['y']:.5f} COUNT={r['COUNT']} dist={r['dist_to_edge_m']:.1f} m")
    table = table.loc[~too_far].reset_index(drop=True)
    info(f"Rows after trim: {len(table)}")
    if len(table) == 0:
        raise RefreshError(
            f"All count points are further than {MAX_SNAP_M} m from the graph. "
            "Check the x/y coordinates in counts.csv are longitude/latitude (EPSG:4326)."
        )

    # Boundary flag for count points.
    boundary_poly = fetch_boundary()
    pts = gpd.GeoDataFrame(table[["x", "y"]].copy(),
                           geometry=gpd.points_from_xy(table["x"], table["y"]), crs="EPSG:4326")
    table["IN_BOUNDARY"] = pts.within(boundary_poly.buffer(0)).to_numpy()
    info(f"Inside {BOUNDARY_OSM_ID}: {int(table['IN_BOUNDARY'].sum())}; "
         f"outside: {int((~table['IN_BOUNDARY']).sum())}")
    return table


# ---------------------------------------------------------------------------
# Step 4. Segment and build design matrices
# ---------------------------------------------------------------------------

def assign_segments(table):
    import numpy as np
    import pandas as pd

    stage("Step 4: Segmenting counts and building design matrices")
    pd_mb = pd.to_numeric(table["PD_MB"], errors="coerce").fillna(0)
    table = table.copy()
    table["SEGMENT"] = np.where(
        table["IN_BOUNDARY"],
        np.where(pd_mb == 0, "CBDP", "CBD"),
        np.where(pd_mb == 0, "SP", "S"),
    )
    counts = table["SEGMENT"].value_counts()
    for seg in SEGMENTS:
        n = int(counts.get(seg, 0))
        info(f"{seg}: {n} count points")
        if n == 0:
            raise RefreshError(
                f"Segment {seg} has no count points. The model cannot be refreshed with this data."
            )
        if n < MIN_ROWS_PER_SEGMENT:
            pause_for_weak_segment(seg, n)
    return table


def build_design(table, segment: str, predictors):
    import pandas as pd
    import statsmodels.api as sm

    seg = table.loc[table["SEGMENT"] == segment].copy()
    for c in predictors + ["COUNT"]:
        if c in seg.columns:
            seg[c] = pd.to_numeric(seg[c], errors="coerce")
    cols = [c for c in predictors if c in seg.columns]
    data = seg[["COUNT"] + cols].dropna()
    cols = [c for c in cols if data[c].nunique() > 1]  # drop zero-variance predictors
    data = data[["COUNT"] + cols]
    y = data["COUNT"]
    X = sm.add_constant(data[cols], has_constant="add")
    return X, y


def build_all_designs(table):
    designs = {}
    for seg in SEGMENTS:
        X, y = build_design(table, seg, SEGMENT_PREDICTORS[seg])
        if len(y) == 0:
            raise RefreshError(
                f"Segment {seg} has no complete rows after removing missing predictor values. "
                "Check the graph attributes were joined (see Step 3 warnings)."
            )
        designs[seg] = (X, y)
    return designs


# ---------------------------------------------------------------------------
# Step 5. Write per-segment inputs
# ---------------------------------------------------------------------------

def write_segment_inputs(out: Path, designs) -> None:
    stage("Step 5: Writing per-segment inputs and correlation matrices")
    for seg, (X, y) in designs.items():
        seg_dir = out / "segments" / seg
        seg_dir.mkdir(parents=True, exist_ok=True)

        export = X.copy()
        export["COUNT"] = y.values
        export.to_csv(seg_dir / "X_y.csv", index=False)

        corr_data = X.drop(columns="const", errors="ignore").copy()
        corr_data["COUNT"] = y.values
        if corr_data.shape[1] >= 2:
            corr_data.corr().round(3).to_csv(seg_dir / "correlation_matrix.csv")
        else:
            warn(f"{seg}: not enough columns for a correlation matrix; skipped.")
        info(f"{seg}: wrote X_y.csv and correlation_matrix.csv")


# ---------------------------------------------------------------------------
# Step 6 and 7. Fit per-segment and combined models
# ---------------------------------------------------------------------------

def fit_segments(out: Path, designs):
    import statsmodels.api as sm

    stage("Step 6: Fitting per-segment models")
    fits = {}
    for seg, (X, y) in designs.items():
        try:
            model = sm.OLS(y, X).fit(maxiter=500, disp=False)
        except Exception as exc:
            raise RefreshError(f"Segment {seg} failed to fit: {exc}") from exc
        (out / "segments" / seg / "regression_summary.txt").write_text(
            model.summary().as_text(), encoding="utf-8")
        fits[seg] = model
        info(f"{seg}: R^2={model.rsquared:.3f}, n={int(model.nobs)}")
    return fits


def fit_combined(out: Path, table):
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm

    stage("Step 7: Fitting combined model")
    all_cols = sorted({c for cols in SEGMENT_PREDICTORS.values() for c in cols} | {"COUNT", "SEGMENT"})
    base = table[all_cols].copy()
    for col in all_cols:
        if col != "SEGMENT":
            base[col] = pd.to_numeric(base[col], errors="coerce")

    ok = pd.Series(True, index=base.index)
    for seg, cols in SEGMENT_PREDICTORS.items():
        in_seg = base["SEGMENT"] == seg
        ok &= ~in_seg | base[cols].notna().all(axis=1)
    base = base.loc[base["SEGMENT"].isin(SEGMENTS) & ok].copy()

    blocks = {}
    for seg in SEGMENTS:
        in_seg = (base["SEGMENT"] == seg).astype(float)
        blocks[f"I_{seg}"] = in_seg
        for col in SEGMENT_PREDICTORS[seg]:
            blocks[f"{col}_{seg}"] = base[col].astype(float) * in_seg

    X = pd.DataFrame(blocks, index=base.index)
    y = base["COUNT"].astype(float)
    keep = [c for c in X.columns if c.startswith("I_") or X[c].nunique() > 1]
    X = X[keep]

    rank = np.linalg.matrix_rank(X.values)
    residual_df = X.shape[0] - rank
    info(f"Combined design: {X.shape}, rank {rank}, residual df {residual_df}")
    if rank < X.shape[1]:
        warn("Combined matrix is rank-deficient; coefficients are not uniquely identified.")
    if residual_df < 10:
        warn(f"Low residual degrees of freedom ({residual_df}); treat coefficients with caution.")

    try:
        model = sm.OLS(y, X).fit()
    except Exception as exc:
        raise RefreshError(f"Combined model failed to fit: {exc}") from exc

    (out / "regression_summary.txt").write_text(model.summary().as_text(), encoding="utf-8")
    info(f"Combined: R^2={model.rsquared:.3f}, n={int(model.nobs)}")

    coef_table = pd.DataFrame({
        "term": model.params.index,
        "coef": model.params.values,
        "std_err": model.bse.values,
        "p_value": model.pvalues.values,
    })
    return model, coef_table


# ---------------------------------------------------------------------------
# Step 8. Predict every edge
# ---------------------------------------------------------------------------

def predict_edges(G, coef_table):
    import pandas as pd
    import networkx as nx
    from shapely.geometry import Point
    from tqdm import tqdm

    stage("Step 8: Predicting every edge (longest step)")
    coef = dict(zip(coef_table["term"], coef_table["coef"].astype(float)))

    # Every predictor term must exist as an attribute on the graph, or predictions would be wrong.
    predictor_bases = {
        term.rpartition("_")[0]
        for term in coef
        if term != "const" and not term.startswith("I_")
    }
    sample = next(iter(G.edges(data=True)))[2]
    missing_terms = sorted(p for p in predictor_bases if p not in sample)
    if missing_terms:
        raise RefreshError(f"Predictor(s) {missing_terms} not found on the graph.\n  {GRAPH_RERUN_MSG}")

    boundary_poly = fetch_boundary()

    def num(data, name):
        try:
            return float(data.get(name, 0) or 0)
        except (TypeError, ValueError):
            return 0.0

    predicted = {}
    segments = {}
    edges_iter = tqdm(G.edges(keys=True, data=True), total=len(G.edges), desc="    predicting", unit="edge")
    for u, v, key, data in edges_iter:
        x_mid = (G.nodes[u]["x"] + G.nodes[v]["x"]) / 2
        y_mid = (G.nodes[u]["y"] + G.nodes[v]["y"]) / 2
        in_boundary = boundary_poly.contains(Point(x_mid, y_mid))
        pd_mb = num(data, "PD_MB")
        if in_boundary:
            seg = "CBDP" if pd_mb == 0 else "CBD"
        else:
            seg = "SP" if pd_mb == 0 else "S"

        value = coef.get(f"I_{seg}", 0.0)
        for term, c in coef.items():
            if term == "const" or term.startswith("I_"):
                continue
            base_name, _, term_seg = term.rpartition("_")
            if term_seg == seg:
                value += c * num(data, base_name)

        predicted[(u, v, key)] = max(value, 0.0)  # clip negatives to 0
        segments[(u, v, key)] = seg

    nx.set_edge_attributes(G, predicted, "PREDICTED_COUNT")
    nx.set_edge_attributes(G, segments, "SEGMENT")

    vals = pd.Series(list(predicted.values()))
    info(f"Edges predicted: {len(predicted)}")
    info(f"PREDICTED_COUNT: min={vals.min():.2f} max={vals.max():.2f} mean={vals.mean():.2f}")


# ---------------------------------------------------------------------------
# Step 9. Export results
# ---------------------------------------------------------------------------

def export_results(G, out: Path, stamp: str):
    import pandas as pd
    import osmnx as ox
    import matplotlib
    matplotlib.use("Agg")  # no display needed
    import matplotlib.pyplot as plt

    stage("Step 9: Exporting GeoPackage and preview PNG")
    _nodes, _edges = ox.graph_to_gdfs(G, nodes=True, edges=True)
    _edges = _edges.reset_index()[["u", "v", "key", "PREDICTED_COUNT", "geometry"]].copy()
    _edges["PREDICTED_COUNT"] = pd.to_numeric(_edges["PREDICTED_COUNT"], errors="coerce").astype(int)

    gpkg_path = out / f"model_{stamp}.gpkg"
    try:
        _edges.to_file(gpkg_path, driver="GPKG", layer="edges_pred")
    except Exception as exc:
        raise RefreshError(
            f"Could not write {gpkg_path.name} ({exc}).\n"
            "  Close it if it is open in QGIS, then run again."
        ) from exc
    info(f"GeoPackage: {gpkg_path.relative_to(REPO_ROOT)}")

    png_path = out / "model_preview.png"
    try:
        fig, ax = plt.subplots(figsize=(12, 12))
        _edges.plot(column="PREDICTED_COUNT", ax=ax, linewidth=0.8, cmap="viridis", legend=True,
                    legend_kwds={"label": "PREDICTED_COUNT", "shrink": 0.5})
        ax.set_title("Predicted cyclist volume")
        ax.set_axis_off()
        fig.savefig(png_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
    except Exception as exc:
        warn(f"Could not write preview PNG ({exc}). GeoPackage was still written.")
    else:
        info(f"Preview: {png_path.relative_to(REPO_ROOT)}")


# ---------------------------------------------------------------------------
# Step 10. Run log and summary
# ---------------------------------------------------------------------------

def write_run_settings(out: Path, started: float, table, stamp: str) -> None:
    stage("Step 10: Writing run settings and summary")
    lines = [
        f"Run timestamp: {stamp}",
        f"Duration (s): {time.time() - started:.1f}",
        "",
        "Inputs",
        f"  counts: {COUNTS_PATH.relative_to(REPO_ROOT)} ({COUNTS_PATH.stat().st_size} bytes)",
        f"  graph:  {GRAPH_PATH.relative_to(REPO_ROOT)} ({GRAPH_PATH.stat().st_size} bytes)",
        "",
        "Settings",
        f"  MAX_SNAP_M = {MAX_SNAP_M}",
        f"  BOUNDARY_OSM_ID = {BOUNDARY_OSM_ID}",
        f"  MIN_ROWS_PER_SEGMENT = {MIN_ROWS_PER_SEGMENT}",
        "",
        "Predictors",
    ]
    for seg in SEGMENTS:
        lines.append(f"  {seg}: {', '.join(SEGMENT_PREDICTORS[seg])}")
    lines += [
        "",
        f"Rows used: {len(table)}",
        "Rows per segment: " + ", ".join(f"{s}={int((table['SEGMENT'] == s).sum())}" for s in SEGMENTS),
        "",
        "Warnings",
    ]
    lines += [f"  - {w}" for w in WARNINGS] or ["  none"]
    (out / "run_settings.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    info(f"Run settings: {(out / 'run_settings.txt').relative_to(REPO_ROOT)}")


def print_final_summary(out: Path) -> None:
    print("\n" + "=" * 60)
    print("Model refresh complete.")
    print(f"Output folder: {out.relative_to(REPO_ROOT)}")
    if WARNINGS:
        print(f"{len(WARNINGS)} warning(s) recorded in run_settings.txt:")
        for w in WARNINGS:
            print(f"  - {w}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    started = time.time()
    stamp = dt.datetime.now().strftime("%y_%m_%d_%H_%M")
    try:
        out = preflight(stamp)
        G = load_graph()
        table = load_counts()
        table = snap_and_trim(table, G)
        table = assign_segments(table)
        designs = build_all_designs(table)
        write_segment_inputs(out, designs)
        fit_segments(out, designs)
        _, coef_table = fit_combined(out, table)
        predict_edges(G, coef_table)
        export_results(G, out, stamp)
        write_run_settings(out, started, table, stamp)
        print_final_summary(out)
        return 0
    except RefreshError as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nStopped by user.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"\nUNEXPECTED ERROR: {exc}", file=sys.stderr)
        print("Details (send this to the maintainer):", file=sys.stderr)
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    sys.exit(main())
