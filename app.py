"""
Antenna Surrogate Interface — 2.45 GHz 2x2 Microstrip Patch Array
=================================================================

Interactive forward predictor for an ML Gaussian-Process surrogate model.
Set 7 geometry parameters with sliders; the app shows predicted antenna
performance *with uncertainty bands*, and — crucially — honestly flags which
predictions are trustworthy (gain/directivity, R^2=0.92) and which are not
(S11 and resonant frequency, R^2<0, because resonance shifts discontinuously).
"""

import logging
import pickle
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import streamlit as st

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PARAM_COLS = ["L", "W", "y0", "Wf", "Wf_qw", "Sx", "Sy"]
TARGETS = ["s11_at_2p45", "s11_min_freq", "max_gain_dbi", "total_efficiency_db"]

# (label, unit, min, max, champion default) — order matches PARAM_COLS exactly.
PARAM_SPEC = {
    "L":     ("Patch length",                   "mm", 38.0, 43.0, 40.5),
    "W":     ("Patch width",                    "mm", 52.0, 65.0, 60.0),
    "y0":    ("Inset feed depth",               "mm", 10.0, 17.0, 14.7),
    "Wf":    ("Feed line width (50 ohm)",       "mm",  4.5,  5.2,  4.85),
    "Wf_qw": ("Quarter-wave transformer width", "mm",  2.2,  2.8,  2.4),
    "Sx":    ("Element spacing (x)",            "mm", 75.0, 95.0, 85.7),
    "Sy":    ("Element spacing (y)",            "mm", 75.0, 95.0, 85.7),
}

# Per-metric trust levels — hard-coded from the model's measured R^2.
# key -> (label, unit, R^2, trust)
TRUST = {
    "max_gain_dbi":        ("Peak Gain / Directivity", "dBi",  0.92, "RELIABLE"),
    "total_efficiency_db": ("Total Efficiency",        "dB",   0.52, "MARGINAL"),
    "s11_at_2p45":         ("S11 at 2.45 GHz",         "dB",  -0.14, "UNRELIABLE"),
    "s11_min_freq":        ("Resonant Frequency",      "GHz", -2.03, "UNRELIABLE"),
}

TRUST_COLOR = {"RELIABLE": "#2e7d32", "MARGINAL": "#f39c12", "UNRELIABLE": "#c0392b"}
TRUST_CAPTION = {
    "RELIABLE":   "Trust this prediction — the surrogate models this well.",
    "MARGINAL":   "Use with caution — moderate agreement with full-wave results.",
    "UNRELIABLE": "Do not trust — verify in full-wave simulation.",
}

# Hard-coded CST-verified champion values (the reference truth for comparison).
CHAMPION_VERIFIED = {
    "s11_at_2p45":  ("S11 at 2.45 GHz",     "-26.4 dB"),
    "s11_min_freq": ("Resonant frequency",  "2.452 GHz"),
    "max_gain_dbi": ("Peak directivity",    "11.03 dBi"),
    "total_efficiency_db": ("Total efficiency", "~99%"),
    "vswr":         ("VSWR",                "1.16"),
    "bw":           ("-10 dB bandwidth",    "36 MHz (2.434-2.470 GHz)"),
}

DATA_PKL = "surrogate_models.pkl"
DATA_CSV = "training_data.csv"


# ---------------------------------------------------------------------------
# Model + data loading
# ---------------------------------------------------------------------------
@st.cache_resource
def load_models():
    """Load-with-fallback (option 3): try the pickle, else refit GPs from CSV."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with open(DATA_PKL, "rb") as f:
                obj = pickle.load(f)
        return obj["models"], obj["scaler"], obj["param_cols"], f"loaded from {DATA_PKL}"
    except Exception as e:
        # Fallback: refit GPs from the CSV (portable, version-proof).
        df = pd.read_csv(DATA_CSV)
        df = df[df["status"] == "ok"] if "status" in df else df
        X = df[PARAM_COLS].values
        scaler = StandardScaler().fit(X)
        Xs = scaler.transform(X)
        models = {}
        for t in TARGETS:
            y = df[t].values
            kernel = (ConstantKernel(1.0) * Matern(nu=2.5, length_scale=[1.0] * X.shape[1])
                      + WhiteKernel(noise_level=0.01))
            gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=15,
                                          normalize_y=True, random_state=42)
            gp.fit(Xs, y)
            models[t] = gp
        return models, scaler, PARAM_COLS, f"refitted from CSV (pickle load failed: {e})"


@st.cache_data
def load_training_data():
    df = pd.read_csv(DATA_CSV)
    df = df[df["status"] == "ok"] if "status" in df else df
    return df


def predict_all(models, scaler, params_dict):
    """Return {target: (mean, std)} — scaling is applied before predict()."""
    x = np.array([[params_dict[c] for c in PARAM_COLS]])   # shape (1,7), raw mm
    xs = scaler.transform(x)                                # MUST scale first
    out = {}
    for name, m in models.items():
        mean, std = m.predict(xs, return_std=True)
        out[name] = (float(mean[0]), float(std[0]))
    return out


# ---------------------------------------------------------------------------
# Geometry sketch
# ---------------------------------------------------------------------------
def draw_geometry(params):
    """Schematic, proportional 2x2 patch layout from current slider values."""
    L, W = params["L"], params["W"]
    Sx, Sy = params["Sx"], params["Sy"]

    fig, ax = plt.subplots(figsize=(4.6, 4.6))

    # Ground plane outline: enclose the array + one patch margin on each side.
    margin_x = W * 0.9
    margin_y = L * 0.9
    gnd_w = Sx + 2 * margin_x
    gnd_h = Sy + 2 * margin_y
    ax.add_patch(Rectangle((-gnd_w / 2, -gnd_h / 2), gnd_w, gnd_h,
                           fill=False, edgecolor="#555", linewidth=1.3, linestyle="--"))

    # Element centres on the Sx-by-Sy grid.
    cx = [-Sx / 2, Sx / 2]
    cy = [-Sy / 2, Sy / 2]
    for x in cx:
        for y in cy:
            ax.add_patch(Rectangle((x - W / 2, y - L / 2), W, L,
                                   facecolor="#c98a2b", edgecolor="#7a4e12", linewidth=1.2))
            # Feed point marker (bottom-centre inset of each patch).
            ax.plot(x, y - L / 2, marker="v", color="#1a1a1a", markersize=6)

    # Dimension annotations for one representative patch (bottom-left).
    x0, y0 = cx[0], cy[0]
    ax.annotate("", xy=(x0 - W / 2, y0 - L / 2 - margin_y * 0.35),
                xytext=(x0 + W / 2, y0 - L / 2 - margin_y * 0.35),
                arrowprops=dict(arrowstyle="<->", color="#333"))
    ax.text(x0, y0 - L / 2 - margin_y * 0.55, f"W = {W:.1f} mm",
            ha="center", va="top", fontsize=8)
    ax.annotate("", xy=(x0 - W / 2 - margin_x * 0.35, y0 - L / 2),
                xytext=(x0 - W / 2 - margin_x * 0.35, y0 + L / 2),
                arrowprops=dict(arrowstyle="<->", color="#333"))
    ax.text(x0 - W / 2 - margin_x * 0.5, y0, f"L = {L:.1f} mm",
            ha="right", va="center", fontsize=8, rotation=90)

    # Spacing annotations (across the array).
    ax.annotate("", xy=(cx[0], -gnd_h / 2 + margin_y * 0.25),
                xytext=(cx[1], -gnd_h / 2 + margin_y * 0.25),
                arrowprops=dict(arrowstyle="<->", color="#0b6fa4"))
    ax.text(0, -gnd_h / 2 + margin_y * 0.10, f"Sx = {Sx:.1f} mm",
            ha="center", va="bottom", fontsize=8, color="#0b6fa4")
    ax.annotate("", xy=(gnd_w / 2 - margin_x * 0.25, cy[0]),
                xytext=(gnd_w / 2 - margin_x * 0.25, cy[1]),
                arrowprops=dict(arrowstyle="<->", color="#0b6fa4"))
    ax.text(gnd_w / 2 - margin_x * 0.10, 0, f"Sy = {Sy:.1f} mm",
            ha="right", va="center", fontsize=8, rotation=90, color="#0b6fa4")

    ax.set_xlim(-gnd_w / 2 * 1.08, gnd_w / 2 * 1.08)
    ax.set_ylim(-gnd_h / 2 * 1.08, gnd_h / 2 * 1.08)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("2x2 patch array layout (schematic, proportional)", fontsize=9)
    fig.tight_layout()
    return fig


def draw_uncertainty(preds):
    """Two-panel uncertainty view across the four metrics.

    Left  : predicted value with its +/- 1 sigma band, in each metric's native units.
    Right : *relative* uncertainty sigma / |value| on a log scale, so S11's huge
            relative uncertainty stays visible even when its absolute sigma is small
            (as at the champion point).

    Hardened: all inputs coerced finite, axis limits pinned to finite margins so
    autoscaling cannot collapse or throw, sigma annotations offset off the markers
    and y-tick labels so nothing overlaps. Always returns a fully-built figure.
    """
    keys = ["max_gain_dbi", "total_efficiency_db", "s11_at_2p45", "s11_min_freq"]
    short = [TRUST[k][0] for k in keys]
    units = [TRUST[k][1] for k in keys]
    colors = [TRUST_COLOR[TRUST[k][3]] for k in keys]

    # Coerce to finite floats so degenerate values can't break autoscaling.
    means = np.array([float(preds[k][0]) for k in keys], dtype=float)
    stds = np.array([max(float(preds[k][1]), 0.0) for k in keys], dtype=float)
    means = np.where(np.isfinite(means), means, 0.0)
    stds = np.where(np.isfinite(stds), stds, 0.0)

    y = np.arange(len(keys))
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(8.4, 3.6))

    # ---- Left: absolute value +/- 1 sigma (native units per metric) ----
    axL.errorbar(means, y, xerr=stds, fmt="none", capsize=6, capthick=2,
                 elinewidth=2.5, ecolor="#888", zorder=2)
    axL.scatter(means, y, c=colors, s=95, zorder=3, edgecolor="#222", linewidth=0.6)
    axL.set_yticks(y)
    axL.set_yticklabels([f"{s}\n({u})" for s, u in zip(short, units)], fontsize=8)
    axL.invert_yaxis()

    lo = float(np.min(means - stds))
    hi = float(np.max(means + stds))
    if not (np.isfinite(lo) and np.isfinite(hi)) or hi <= lo:
        lo, hi = float(means.min()) - 1.0, float(means.max()) + 1.0
        if hi <= lo:
            lo, hi = lo - 1.0, hi + 1.0
    pad = max((hi - lo) * 0.18, 0.5)
    axL.set_xlim(lo - pad, hi + pad)

    # sigma annotation below each marker so it never sits on the point/labels.
    for yi, m, s, c in zip(y, means, stds, colors):
        axL.annotate(f"±{s:.3g}", (m, yi), textcoords="offset points",
                     xytext=(0, -13), ha="center", va="top", fontsize=7.5, color=c)
    axL.set_xlabel("Predicted value (native units)")
    axL.set_title("Absolute prediction  ± 1σ", fontsize=9)
    axL.grid(axis="x", alpha=0.3)

    # ---- Right: relative uncertainty sigma / |value| (log scale) ----
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.where(np.abs(means) > 1e-9, stds / np.abs(means), np.nan)
    floor = 1e-4  # display floor so log bars for near-zero sigma stay visible
    rel_plot = np.where(np.isfinite(rel) & (rel > floor), rel, floor)
    axR.barh(y, rel_plot, color=colors, alpha=0.85, edgecolor="#222", linewidth=0.6)
    axR.set_xscale("log")
    axR.set_yticks(y)
    axR.set_yticklabels([])
    axR.invert_yaxis()
    top = max(1.0, float(np.nanmax(rel_plot)) * 3.0)
    if not np.isfinite(top) or top <= floor:
        top = 1.0
    axR.set_xlim(floor, top)
    for yi, r, rp in zip(y, rel, rel_plot):
        txt = "n/a" if not np.isfinite(r) else f"{r * 100:.2g}%"
        axR.annotate(txt, (rp, yi), textcoords="offset points", xytext=(4, 0),
                     ha="left", va="center", fontsize=7.5, color="#333")
    axR.set_xlabel("Relative uncertainty  σ / |value|  (log)")
    axR.set_title("Relative uncertainty (σ / |value|)", fontsize=9)
    axR.grid(axis="x", alpha=0.3, which="both")

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
def main():
    st.set_page_config(page_title="Antenna Surrogate Interface",
                       layout="wide", page_icon="📡")

    # Guard: required data files.
    import os
    if not os.path.exists(DATA_PKL) and not os.path.exists(DATA_CSV):
        st.error(
            f"Missing data files. Place **{DATA_PKL}** and **{DATA_CSV}** in the "
            f"same folder as `app.py` before running.")
        st.stop()

    try:
        models, scaler, param_cols, source = load_models()
    except Exception as e:
        st.error(f"Could not load or refit the surrogate model: {e}")
        st.stop()

    try:
        train = load_training_data()
    except Exception as e:
        st.error(f"Could not load {DATA_CSV}: {e}")
        st.stop()

    n_train = len(train)

    # Initialise slider state.
    for key, (_, _, lo, hi, dflt) in PARAM_SPEC.items():
        st.session_state.setdefault(key, dflt)

    if st.session_state.get("_do_reset"):
        for key, (_, _, _, _, dflt) in PARAM_SPEC.items():
            st.session_state[key] = dflt
        st.session_state["_do_reset"] = False

    st.title("📡 Antenna Surrogate Interface")
    st.caption("Forward predictor for a 2.45 GHz 2x2 microstrip patch antenna array "
               "— Gaussian-Process ML surrogate with honest uncertainty.")

    # How-to-read banner — visible on first load, before any slider moves.
    st.info(
        "**How to read this tool:** gain / directivity is **trustworthy** (R²=0.92); "
        "**S11 and resonant frequency are NOT** (R²<0) and must be verified in "
        "full-wave simulation.",
        icon="🧭")

    # --------------------- SIDEBAR: geometry inputs -------------------------
    # Inputs live in the sidebar so the results summary stays at the top of the
    # page on narrow / mobile screens instead of below all 7 sliders.
    with st.sidebar:
        st.subheader("Geometry inputs")
        st.markdown("Set the 7 design parameters; predictions update live.")

        def _reset():
            st.session_state["_do_reset"] = True

        st.button("↺ Reset to champion design", on_click=_reset, use_container_width=True)

        params = {}
        for key in PARAM_COLS:
            label, unit, lo, hi, dflt = PARAM_SPEC[key]
            # value comes from st.session_state[key] (pre-seeded via setdefault /
            # reset), so we pass only key= to avoid a session-state widget warning.
            params[key] = st.slider(
                f"{label} — {key} ({unit})", min_value=lo, max_value=hi,
                step=0.1, key=key)

        st.caption(f"Model trained on {n_train} full-wave CST simulations.")

        # In-range / out-of-range indicator vs training envelope.
        out_of_range = []
        for key in PARAM_COLS:
            col_min, col_max = float(train[key].min()), float(train[key].max())
            if params[key] < col_min or params[key] > col_max:
                out_of_range.append(key)
        if out_of_range:
            st.warning(
                "⚠ Some parameters are outside the training envelope "
                f"({', '.join(out_of_range)}) — predictions (especially S11) are "
                "extrapolated and less trustworthy. This is *why* the uncertainty "
                "bands grow.")
        else:
            st.success("✅ All parameters are within the training envelope.")

        with st.expander("About this model"):
            st.markdown(
                f"""
This surrogate was trained on **{n_train} full-wave CST simulations** sampled by
Latin Hypercube Sampling. It predicts **gain / directivity accurately (R²=0.92)**,
but the **reflection coefficient (S11) and resonant frequency are NOT reliably
predictable (R²<0)** because resonance shifts *discontinuously* between modes.

The uncertainty bands make this visible: gain has a tiny error bar, S11 has a huge
one. **Any S11-critical design must be verified in full-wave simulation.**

Note: the champion design was reached by **manual tuning + CST verification**, not
by the surrogate.

_Model source: {source}._
""")

    # ---------------------- MAIN: predictions -------------------------------
    preds = predict_all(models, scaler, params)

    st.subheader("Predicted performance")

    # Four metric cards in a 2x2 grid.
    card_keys = ["max_gain_dbi", "total_efficiency_db", "s11_at_2p45", "s11_min_freq"]
    rows = [card_keys[:2], card_keys[2:]]
    for row in rows:
        cols = st.columns(2)
        for col, key in zip(cols, row):
            label, unit, r2, trust = TRUST[key]
            mean, std = preds[key]
            color = TRUST_COLOR[trust]
            with col:
                if trust == "UNRELIABLE":
                    # De-emphasise: a hurried reader must not take this at face value.
                    value_html = (
                        f'<div style="font-size:1.3rem;font-weight:600;color:#9aa0a6;">'
                        f'≈ {mean:.2f} ± {std:.2g} {unit}'
                        f'<span style="font-size:0.7rem;font-weight:600;"> (unverified)'
                        f'</span></div>')
                else:
                    value_html = (
                        f'<div style="font-size:1.5rem;font-weight:700;">'
                        f'{mean:.2f} ± {std:.2g} {unit}</div>')
                st.markdown(
                    f"""
<div style="border:1px solid #ddd;border-left:6px solid {color};
     border-radius:8px;padding:12px 14px;margin-bottom:6px;">
  <div style="font-size:0.85rem;color:#666;">{label}</div>
  {value_html}
  <div style="display:inline-block;background:{color};color:white;
       border-radius:4px;padding:1px 8px;font-size:0.75rem;font-weight:600;">
       {trust} · R²={r2:+.2f}</div>
</div>
""", unsafe_allow_html=True)

                if key == "s11_at_2p45":
                    passes = mean <= -10.0
                    verdict = "PASS" if passes else "FAIL"
                    op = "≤" if passes else ">"
                    st.caption(f"−10 dB match threshold (on an *unverified* value): "
                               f"**{verdict}** ({mean:.2f} dB {op} −10 dB)")

                if trust == "UNRELIABLE":
                    st.markdown(
                        f"<div style='color:{TRUST_COLOR['UNRELIABLE']};"
                        f"font-size:0.8rem;'>⚠ The surrogate cannot reliably "
                        f"predict this — verify in full-wave simulation.</div>",
                        unsafe_allow_html=True)
                else:
                    st.caption(TRUST_CAPTION[trust])

    st.markdown("---")
    st.markdown("#### Uncertainty across metrics")
    try:
        ufig = draw_uncertainty(preds)
        st.pyplot(ufig)
        plt.close(ufig)          # avoid stale figure/state reuse across reruns
        st.caption(
            "Left: each metric's predicted value with its ±1σ band, in native units. "
            "Right: relative uncertainty σ/|value| on a log scale, so S11's large "
            "relative uncertainty stays visible even when its absolute σ looks small.")
    except Exception:
        logging.exception("Uncertainty chart generation failed")
        st.warning("Uncertainty chart could not be drawn for these inputs.")

    st.markdown("#### Live geometry sketch")
    try:
        gfig = draw_geometry(params)
        st.pyplot(gfig)
        plt.close(gfig)
        st.caption(
            "Schematic 2×2 patch array (proportional): four W×L patches on an "
            "Sx-by-Sy grid inside the ground-plane outline, feed points marked. "
            "Updates live with the sliders.")
    except Exception:
        logging.exception("Geometry sketch generation failed")
        st.warning("Geometry sketch could not be drawn for these inputs.")

    # ------------------ Predicted vs. Verified Champion table ---------------
    st.markdown("---")
    st.subheader("Predicted vs. Verified Champion")
    st.caption("Model prediction at the champion geometry vs. the CST-verified truth. "
               "Some verified quantities (VSWR, bandwidth) are not predicted by the "
               "surrogate and are shown for reference only.")

    champ_params = {k: PARAM_SPEC[k][4] for k in PARAM_COLS}
    champ_preds = predict_all(models, scaler, champ_params)

    def fmt_pred(key):
        m, s = champ_preds[key]
        unit = TRUST[key][1]
        return f"{m:.2f} ± {s:.2g} {unit}"

    table = pd.DataFrame([
        {"Metric": CHAMPION_VERIFIED["s11_at_2p45"][0],
         "Surrogate prediction": fmt_pred("s11_at_2p45"),
         "CST-verified": CHAMPION_VERIFIED["s11_at_2p45"][1],
         "Trust": TRUST["s11_at_2p45"][3]},
        {"Metric": CHAMPION_VERIFIED["s11_min_freq"][0],
         "Surrogate prediction": fmt_pred("s11_min_freq"),
         "CST-verified": CHAMPION_VERIFIED["s11_min_freq"][1],
         "Trust": TRUST["s11_min_freq"][3]},
        {"Metric": CHAMPION_VERIFIED["max_gain_dbi"][0],
         "Surrogate prediction": fmt_pred("max_gain_dbi"),
         "CST-verified": CHAMPION_VERIFIED["max_gain_dbi"][1],
         "Trust": TRUST["max_gain_dbi"][3]},
        {"Metric": CHAMPION_VERIFIED["total_efficiency_db"][0],
         "Surrogate prediction": fmt_pred("total_efficiency_db"),
         "CST-verified": CHAMPION_VERIFIED["total_efficiency_db"][1],
         "Trust": TRUST["total_efficiency_db"][3]},
        {"Metric": CHAMPION_VERIFIED["vswr"][0],
         "Surrogate prediction": "— (not modelled)",
         "CST-verified": CHAMPION_VERIFIED["vswr"][1], "Trust": "—"},
        {"Metric": CHAMPION_VERIFIED["bw"][0],
         "Surrogate prediction": "— (not modelled)",
         "CST-verified": CHAMPION_VERIFIED["bw"][1], "Trust": "—"},
    ])
    def _trust_col_style(col):
        return [
            f"background-color:{TRUST_COLOR[v]};color:white;font-weight:600;"
            "text-align:center;" if v in TRUST_COLOR else ""
            for v in col
        ]

    styler = table.style.apply(_trust_col_style, subset=["Trust"])
    st.dataframe(styler, hide_index=True, use_container_width=True)


if __name__ == "__main__":
    main()
