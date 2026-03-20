"""
Seizure Prediction — Professional Streamlit Dashboard
Neural Guard: Real-time pre-ictal EEG analysis using Temporal GNN
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
import os
import torch
from pathlib import Path

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="Neural Guard — Seizure Prediction",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
    --bg-dark:     #0a0e1a;
    --bg-card:     #111827;
    --bg-card2:    #1a2234;
    --accent:      #00d4aa;
    --accent2:     #7c6ff7;
    --danger:      #ef4444;
    --warning:     #f59e0b;
    --text-pri:    #f0f4ff;
    --text-sec:    #8b9cc8;
    --border:      rgba(255,255,255,0.07);
}

html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--bg-dark) !important;
    font-family: 'DM Sans', sans-serif;
    color: var(--text-pri);
}

[data-testid="stSidebar"] {
    background-color: var(--bg-card) !important;
    border-right: 1px solid var(--border);
}

[data-testid="stSidebar"] * { color: var(--text-pri) !important; }

h1, h2, h3 { font-family: 'Space Mono', monospace !important; }

.metric-card {
    background: var(--bg-card2);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 12px;
    transition: border-color 0.3s;
}
.metric-card:hover { border-color: var(--accent); }

.metric-value {
    font-family: 'Space Mono', monospace;
    font-size: 2.2rem;
    font-weight: 700;
    color: var(--accent);
    line-height: 1;
    margin: 6px 0 2px;
}
.metric-label {
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: var(--text-sec);
}
.metric-sub {
    font-size: 0.82rem;
    color: var(--text-sec);
    margin-top: 4px;
}

.alert-box {
    border-radius: 10px;
    padding: 16px 20px;
    margin: 12px 0;
    font-family: 'Space Mono', monospace;
    font-size: 0.85rem;
    display: flex;
    align-items: center;
    gap: 12px;
}
.alert-danger  { background: rgba(239,68,68,0.12);  border: 1px solid rgba(239,68,68,0.4);  color: #fca5a5; }
.alert-warning { background: rgba(245,158,11,0.12); border: 1px solid rgba(245,158,11,0.4); color: #fcd34d; }
.alert-safe    { background: rgba(0,212,170,0.10);  border: 1px solid rgba(0,212,170,0.35); color: #6ee7d4; }

.section-header {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--text-sec);
    padding: 6px 0 10px;
    border-bottom: 1px solid var(--border);
    margin-bottom: 16px;
}

.pill {
    display: inline-block;
    padding: 3px 12px;
    border-radius: 999px;
    font-size: 0.72rem;
    font-family: 'Space Mono', monospace;
    letter-spacing: 0.05em;
}
.pill-green  { background: rgba(0,212,170,0.15); color: var(--accent); border: 1px solid rgba(0,212,170,0.3); }
.pill-red    { background: rgba(239,68,68,0.15);  color: #f87171;       border: 1px solid rgba(239,68,68,0.3); }
.pill-purple { background: rgba(124,111,247,0.15);color: #a78bfa;       border: 1px solid rgba(124,111,247,0.3);}

stButton > button {
    background: var(--accent) !important;
    color: #000 !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.8rem !important;
}

[data-testid="stFileUploader"] {
    background: var(--bg-card2) !important;
    border: 1px dashed rgba(0,212,170,0.3) !important;
    border-radius: 10px !important;
}

[data-testid="stSlider"] > div { color: var(--accent) !important; }

div[data-testid="metric-container"] {
    background: var(--bg-card2);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 14px;
}
</style>
""", unsafe_allow_html=True)


# ── Helper: load model ────────────────────────────────────────────────────────
@st.cache_resource
def load_model(model_path: str):
    try:
        from model import SeizureTGNN
        model = SeizureTGNN(n_channels=23, n_samples=7680)
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        model.eval()
        return model, None
    except FileNotFoundError:
        return None, "model_not_found"
    except Exception as e:
        return None, str(e)


# ── Helper: simulate or run real prediction ───────────────────────────────────
def run_prediction(edf_path: str, model, threshold: float, n_windows: int = 30):
    """
    If model is loaded: runs real preprocessed prediction.
    If no model: generates realistic synthetic demo data.
    Returns list of dicts: [{window, prob, alert, channels}]
    """
    if model is not None:
        try:
            from preprocess import preprocess_edf
            from graph_builder import epoch_to_graph
            from torch_geometric.data import Batch

            epochs = preprocess_edf(edf_path)[:n_windows]
            results = []
            for i, epoch in enumerate(epochs):
                graph = epoch_to_graph(epoch, label=0)
                batch = Batch.from_data_list([graph])
                with torch.no_grad():
                    logits = model(batch)
                    prob = float(torch.softmax(logits, dim=1)[0, 1])
                results.append({
                    "window": i,
                    "prob": prob,
                    "alert": prob >= threshold,
                    "epoch": epoch,
                })
            return results, None
        except Exception as e:
            return None, str(e)

    # ── Demo mode: synthetic realistic EEG trajectory ──
    np.random.seed(42)
    results = []
    for i in range(n_windows):
        # Simulate pre-ictal buildup in last 30% of windows
        if i < n_windows * 0.55:
            base = 0.08 + np.random.normal(0, 0.04)
        elif i < n_windows * 0.75:
            base = 0.18 + (i / n_windows) * 0.35 + np.random.normal(0, 0.06)
        else:
            base = 0.55 + (i / n_windows) * 0.4 + np.random.normal(0, 0.05)
        prob = float(np.clip(base, 0.01, 0.99))
        results.append({
            "window": i,
            "prob": prob,
            "alert": prob >= threshold,
            "epoch": np.random.randn(23, 7680).astype(np.float32),
        })
    return results, None


# ── Plotly theme shared settings ──────────────────────────────────────────────
PLOT_BG   = "#111827"
PAPER_BG  = "#111827"
GRID_COL  = "rgba(255,255,255,0.05)"
FONT_FAM  = "DM Sans, sans-serif"
MONO_FAM  = "Space Mono, monospace"


def make_prob_chart(results, threshold):
    windows = [r["window"] for r in results]
    probs   = [r["prob"]   for r in results]
    alerts  = [r["alert"]  for r in results]

    colors = ["#ef4444" if a else ("#f59e0b" if p > threshold * 0.75 else "#00d4aa")
              for p, a in zip(probs, alerts)]

    fig = go.Figure()

    # Threshold line
    fig.add_hline(y=threshold, line_dash="dash",
                  line_color="rgba(245,158,11,0.6)", line_width=1.5,
                  annotation_text=f"Alert threshold ({threshold:.0%})",
                  annotation_font_color="#fcd34d",
                  annotation_font_size=11)

    # Danger zone fill
    fig.add_hrect(y0=threshold, y1=1.0,
                  fillcolor="rgba(239,68,68,0.06)",
                  line_width=0)

    # Area fill
    fig.add_trace(go.Scatter(
        x=windows, y=probs,
        fill="tozeroy",
        fillcolor="rgba(0,212,170,0.07)",
        line=dict(color="rgba(0,212,170,0.0)"),
        showlegend=False,
        hoverinfo="skip",
    ))

    # Probability line
    fig.add_trace(go.Scatter(
        x=windows, y=probs,
        mode="lines+markers",
        name="Pre-ictal probability",
        line=dict(color="#00d4aa", width=2),
        marker=dict(color=colors, size=7, line=dict(width=0)),
        hovertemplate="Window %{x}<br>Probability: %{y:.1%}<extra></extra>",
    ))

    fig.update_layout(
        plot_bgcolor=PLOT_BG, paper_bgcolor=PAPER_BG,
        font=dict(family=FONT_FAM, color="#8b9cc8"),
        margin=dict(l=10, r=10, t=20, b=10),
        height=260,
        xaxis=dict(title="Window index", gridcolor=GRID_COL, showgrid=True,
                   zeroline=False, title_font_color="#8b9cc8"),
        yaxis=dict(title="P(pre-ictal)", gridcolor=GRID_COL, showgrid=True,
                   range=[0, 1.05], tickformat=".0%", zeroline=False,
                   title_font_color="#8b9cc8"),
        showlegend=False,
    )
    return fig


def make_eeg_chart(epoch: np.ndarray, n_ch: int = 8):
    """Plot first n_ch EEG channels stacked."""
    ch_names = ["FP1","FP2","F7","F3","FZ","F4","F8","T7"][:n_ch]
    t = np.linspace(0, 30, epoch.shape[1])
    fig = go.Figure()
    offset_scale = 3.0

    for i, name in enumerate(ch_names):
        signal = epoch[i] + i * offset_scale
        fig.add_trace(go.Scatter(
            x=t, y=signal,
            mode="lines",
            name=name,
            line=dict(width=0.8,
                      color=px.colors.sequential.Teal[2 + i % 6]),
            hoverinfo="skip",
        ))
        fig.add_annotation(x=-0.5, y=i * offset_scale,
                           text=name, showarrow=False,
                           font=dict(size=10, color="#8b9cc8", family=MONO_FAM),
                           xanchor="right")

    fig.update_layout(
        plot_bgcolor=PLOT_BG, paper_bgcolor=PAPER_BG,
        font=dict(family=FONT_FAM, color="#8b9cc8"),
        margin=dict(l=55, r=10, t=10, b=30),
        height=300,
        xaxis=dict(title="Time (s)", gridcolor=GRID_COL, zeroline=False,
                   title_font_color="#8b9cc8"),
        yaxis=dict(showticklabels=False, gridcolor=GRID_COL, zeroline=False),
        showlegend=False,
    )
    return fig


def make_brain_graph(epoch: np.ndarray, threshold: float = 0.35):
    """Render top-down 2D brain connectivity map."""
    from itertools import combinations

    CH_POS = {
        "FP1": (-0.3, 0.9), "FP2": (0.3, 0.9),
        "F7":  (-0.7, 0.5), "F3":  (-0.4, 0.6), "FZ": (0.0, 0.7),
        "F4":  (0.4, 0.6),  "F8":  (0.7, 0.5),
        "T7":  (-1.0, 0.0), "C3":  (-0.6, 0.0), "CZ": (0.0, 0.0),
        "C4":  (0.6, 0.0),  "T8":  (1.0, 0.0),
        "P7":  (-0.7,-0.5), "P3":  (-0.4,-0.6), "PZ": (0.0,-0.7),
        "P4":  (0.4,-0.6),  "P8":  (0.7,-0.5),
        "O1":  (-0.3,-0.9), "O2":  (0.3,-0.9),
        "F1":  (-0.2, 0.65),"F2":  (0.2, 0.65),
        "FC1": (-0.35,0.35),"FC2": (0.35,0.35),
    }
    names = list(CH_POS.keys())[:23]
    xs    = [CH_POS[n][0] for n in names]
    ys    = [CH_POS[n][1] for n in names]

    # Compute mean power per channel → node color
    power = np.mean(epoch ** 2, axis=1)
    power_norm = (power - power.min()) / (power.max() - power.min() + 1e-8)

    fig = go.Figure()

    # Edges (PLV approximated from correlation for speed)
    for i, j in combinations(range(len(names)), 2):
        corr = float(np.abs(np.corrcoef(epoch[i], epoch[j])[0, 1]))
        if corr > threshold:
            alpha = corr * 0.5
            fig.add_trace(go.Scatter(
                x=[xs[i], xs[j], None],
                y=[ys[i], ys[j], None],
                mode="lines",
                line=dict(width=corr * 2.5,
                          color=f"rgba(124,111,247,{alpha:.2f})"),
                hoverinfo="skip", showlegend=False,
            ))

    # Head outline
    theta = np.linspace(0, 2 * np.pi, 120)
    fig.add_trace(go.Scatter(
        x=np.cos(theta), y=np.sin(theta),
        mode="lines",
        line=dict(color="rgba(255,255,255,0.12)", width=1.5),
        hoverinfo="skip", showlegend=False,
    ))

    # Nodes
    fig.add_trace(go.Scatter(
        x=xs, y=ys,
        mode="markers+text",
        text=names,
        textposition="top center",
        textfont=dict(size=9, color="#8b9cc8", family=MONO_FAM),
        marker=dict(
            size=16,
            color=power_norm,
            colorscale=[[0, "#1a2234"], [0.5, "#7c6ff7"], [1, "#00d4aa"]],
            showscale=True,
            colorbar=dict(
                title="Power",
                thickness=10,
                tickfont=dict(color="#8b9cc8", size=9),
                titlefont=dict(color="#8b9cc8", size=10),
            ),
            line=dict(color="rgba(255,255,255,0.2)", width=1),
        ),
        hovertemplate="%{text}<br>Power: %{marker.color:.3f}<extra></extra>",
        showlegend=False,
    ))

    fig.update_layout(
        plot_bgcolor=PLOT_BG, paper_bgcolor=PAPER_BG,
        margin=dict(l=10, r=10, t=10, b=10),
        height=320,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                   range=[-1.25, 1.25]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                   range=[-1.15, 1.15], scaleanchor="x"),
    )
    return fig


def make_confidence_gauge(prob: float):
    color = "#ef4444" if prob > 0.75 else ("#f59e0b" if prob > 0.45 else "#00d4aa")
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(prob * 100, 1),
        number=dict(suffix="%", font=dict(color=color, family=MONO_FAM, size=32)),
        gauge=dict(
            axis=dict(range=[0, 100], tickwidth=0,
                      tickcolor="transparent",
                      tickfont=dict(color="#8b9cc8", size=9)),
            bar=dict(color=color, thickness=0.25),
            bgcolor=PLOT_BG,
            borderwidth=0,
            steps=[
                dict(range=[0, 45],  color="rgba(0,212,170,0.1)"),
                dict(range=[45, 75], color="rgba(245,158,11,0.1)"),
                dict(range=[75, 100],color="rgba(239,68,68,0.1)"),
            ],
            threshold=dict(line=dict(color="rgba(255,255,255,0.3)", width=2),
                           thickness=0.75, value=75),
        ),
    ))
    fig.update_layout(
        plot_bgcolor=PLOT_BG, paper_bgcolor=PAPER_BG,
        margin=dict(l=20, r=20, t=30, b=10),
        height=200,
        font=dict(color="#8b9cc8"),
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='padding: 8px 0 20px;'>
      <div style='font-family: Space Mono, monospace; font-size: 1.1rem;
                  color: #00d4aa; letter-spacing: 0.08em;'>NEURAL GUARD</div>
      <div style='font-size: 0.72rem; color: #8b9cc8; letter-spacing: 0.15em;
                  text-transform: uppercase; margin-top: 2px;'>
        Pre-ictal EEG Analysis v1.0</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">Input</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload EEG file (.edf)",
        type=["edf"],
        help="CHB-MIT format .edf files supported"
    )

    use_demo = st.checkbox("Use demo data (no file needed)", value=True)

    st.markdown('<div class="section-header">Model</div>', unsafe_allow_html=True)

    model_path = st.text_input(
        "Model checkpoint path",
        value="best_model.pt",
        help="Path to your trained .pt file"
    )

    st.markdown('<div class="section-header">Parameters</div>', unsafe_allow_html=True)

    threshold = st.slider(
        "Alert threshold", 0.40, 0.95, 0.75, 0.05,
        help="Probability above which an alert is triggered"
    )

    n_windows = st.slider(
        "Windows to analyze", 10, 60, 30, 5,
        help="Number of 30-second EEG epochs to process"
    )

    horizon = st.slider(
        "Sustained alert horizon (windows)", 2, 8, 5,
        help="How many consecutive high-prob windows before final alert"
    )

    st.markdown('<div class="section-header">Display</div>', unsafe_allow_html=True)
    show_raw     = st.checkbox("Show raw EEG waveform", value=True)
    show_graph   = st.checkbox("Show brain connectivity map", value=True)
    animate_scan = st.checkbox("Animate scan on run", value=True)

    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.7rem; color: #8b9cc8; line-height: 1.7;'>
    Dataset: CHB-MIT Scalp EEG<br>
    Architecture: Temporal GNN (STGCN)<br>
    Edge weights: Phase Locking Value<br>
    Focal Loss · Subject-adaptive
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN LAYOUT
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div style='padding: 10px 0 4px;'>
  <span style='font-family: Space Mono, monospace; font-size: 1.6rem;
               color: #f0f4ff; letter-spacing: -0.01em;'>
    Neural Guard
  </span>
  <span style='font-family: Space Mono, monospace; font-size: 1.6rem;
               color: #00d4aa;'> /</span>
  <span style='font-family: DM Sans, sans-serif; font-size: 1rem;
               color: #8b9cc8; margin-left: 8px;'>
    Pre-seizure prediction dashboard
  </span>
</div>
""", unsafe_allow_html=True)

# ── Top status bar ────────────────────────────────────────────────────────────
model, model_err = load_model(model_path)
col_s1, col_s2, col_s3, col_s4 = st.columns(4)

with col_s1:
    if model:
        st.markdown('<span class="pill pill-green">Model loaded</span>',
                    unsafe_allow_html=True)
    else:
        st.markdown('<span class="pill pill-purple">Demo mode</span>',
                    unsafe_allow_html=True)
with col_s2:
    st.markdown(f'<span class="pill pill-green">Threshold: {threshold:.0%}</span>',
                unsafe_allow_html=True)
with col_s3:
    st.markdown(f'<span class="pill pill-purple">Windows: {n_windows}</span>',
                unsafe_allow_html=True)
with col_s4:
    st.markdown(f'<span class="pill pill-purple">Horizon: {horizon}</span>',
                unsafe_allow_html=True)

st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

# ── Run button ────────────────────────────────────────────────────────────────
run_col, _ = st.columns([1, 3])
with run_col:
    run_btn = st.button("Run analysis", use_container_width=True)

if run_btn or "results" in st.session_state:

    if run_btn:
        # Determine input path
        edf_path = None
        if uploaded_file and not use_demo:
            tmp_path = Path("tmp_upload.edf")
            tmp_path.write_bytes(uploaded_file.read())
            edf_path = str(tmp_path)
        elif not use_demo:
            st.warning("No file uploaded — switching to demo mode.")

        # Animate scan
        if animate_scan:
            prog = st.progress(0, text="Preprocessing EEG signal...")
            for i in range(1, 101):
                time.sleep(0.012)
                label = ("Preprocessing EEG signal..."   if i < 30 else
                         "Building brain graphs (PLV)..."  if i < 55 else
                         "Running Temporal GNN..."         if i < 80 else
                         "Generating predictions...")
                prog.progress(i, text=label)
            prog.empty()

        results, err = run_prediction(edf_path or "demo", model, threshold, n_windows)
        if err:
            st.error(f"Prediction error: {err}")
            st.stop()

        st.session_state["results"] = results

    results = st.session_state.get("results", [])
    if not results:
        st.stop()

    # ── Alert status ──────────────────────────────────────────────────────────
    probs      = [r["prob"] for r in results]
    alerts     = [r["alert"] for r in results]
    max_prob   = max(probs)
    mean_prob  = np.mean(probs)
    n_alerts   = sum(alerts)
    sustained  = sum(1 for i in range(len(alerts) - horizon + 1)
                     if all(alerts[i:i+horizon])) > 0

    if sustained:
        st.markdown(f"""
        <div class="alert-box alert-danger">
          <span style='font-size:1.2rem'>⚠</span>
          <span>SEIZURE ALERT — Sustained pre-ictal activity detected over {horizon}+ consecutive windows.
          Peak probability: {max_prob:.1%}</span>
        </div>""", unsafe_allow_html=True)
    elif n_alerts > 0:
        st.markdown(f"""
        <div class="alert-box alert-warning">
          <span style='font-size:1.2rem'>△</span>
          <span>WARNING — {n_alerts} high-probability window(s) detected.
          Monitoring for sustained activity. Peak: {max_prob:.1%}</span>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="alert-box alert-safe">
          <span style='font-size:1.2rem'>✓</span>
          <span>NORMAL — No pre-ictal activity detected. Max probability: {max_prob:.1%}</span>
        </div>""", unsafe_allow_html=True)

    # ── Metrics row ───────────────────────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)

    with m1:
        st.markdown(f"""
        <div class="metric-card">
          <div class="metric-label">Peak probability</div>
          <div class="metric-value" style="color:{'#ef4444' if max_prob>threshold else '#00d4aa'}">
            {max_prob:.1%}</div>
          <div class="metric-sub">Highest window score</div>
        </div>""", unsafe_allow_html=True)

    with m2:
        st.markdown(f"""
        <div class="metric-card">
          <div class="metric-label">Mean probability</div>
          <div class="metric-value">{mean_prob:.1%}</div>
          <div class="metric-sub">Across {len(results)} windows</div>
        </div>""", unsafe_allow_html=True)

    with m3:
        st.markdown(f"""
        <div class="metric-card">
          <div class="metric-label">Alert windows</div>
          <div class="metric-value" style="color:{'#ef4444' if n_alerts>0 else '#00d4aa'}">
            {n_alerts}</div>
          <div class="metric-sub">Above {threshold:.0%} threshold</div>
        </div>""", unsafe_allow_html=True)

    with m4:
        status_text  = "ALERT"    if sustained else ("WARNING" if n_alerts > 0 else "SAFE")
        status_color = "#ef4444"  if sustained else ("#f59e0b" if n_alerts > 0 else "#00d4aa")
        st.markdown(f"""
        <div class="metric-card">
          <div class="metric-label">Overall status</div>
          <div class="metric-value" style="color:{status_color}; font-size:1.6rem;">
            {status_text}</div>
          <div class="metric-sub">Horizon: {horizon} windows</div>
        </div>""", unsafe_allow_html=True)

    # ── Main charts ───────────────────────────────────────────────────────────
    left_col, right_col = st.columns([2, 1])

    with left_col:
        st.markdown('<div class="section-header">Pre-ictal probability over time</div>',
                    unsafe_allow_html=True)
        st.plotly_chart(make_prob_chart(results, threshold),
                        use_container_width=True, config={"displayModeBar": False})

    with right_col:
        st.markdown('<div class="section-header">Current window risk</div>',
                    unsafe_allow_html=True)
        latest_prob = results[-1]["prob"]
        st.plotly_chart(make_confidence_gauge(latest_prob),
                        use_container_width=True, config={"displayModeBar": False})

        risk_level = ("Critical — immediate attention"  if latest_prob > 0.75 else
                      "Elevated — continue monitoring"   if latest_prob > 0.45 else
                      "Low — within normal range")
        risk_color = ("#fca5a5" if latest_prob > 0.75 else
                      "#fcd34d" if latest_prob > 0.45 else "#6ee7d4")
        st.markdown(f"""
        <div style='text-align:center; font-family: Space Mono, monospace;
                    font-size:0.75rem; color:{risk_color}; margin-top: -8px;'>
          {risk_level}
        </div>""", unsafe_allow_html=True)

    # ── EEG waveform ─────────────────────────────────────────────────────────
    if show_raw:
        st.markdown('<div class="section-header">Raw EEG — last analyzed window</div>',
                    unsafe_allow_html=True)
        last_epoch = results[-1]["epoch"]
        st.plotly_chart(make_eeg_chart(last_epoch),
                        use_container_width=True, config={"displayModeBar": False})

    # ── Brain connectivity ────────────────────────────────────────────────────
    if show_graph:
        g1, g2 = st.columns([1, 1])
        with g1:
            st.markdown('<div class="section-header">Brain connectivity map</div>',
                        unsafe_allow_html=True)
            st.plotly_chart(make_brain_graph(results[-1]["epoch"]),
                            use_container_width=True, config={"displayModeBar": False})
        with g2:
            st.markdown('<div class="section-header">Probability distribution</div>',
                        unsafe_allow_html=True)
            fig_hist = go.Figure(go.Histogram(
                x=probs, nbinsx=15,
                marker_color="#7c6ff7",
                marker_line_color="#111827",
                marker_line_width=1,
            ))
            fig_hist.add_vline(x=threshold, line_dash="dash",
                               line_color="#f59e0b", line_width=1.5)
            fig_hist.update_layout(
                plot_bgcolor=PLOT_BG, paper_bgcolor=PAPER_BG,
                font=dict(family=FONT_FAM, color="#8b9cc8"),
                margin=dict(l=10, r=10, t=10, b=10),
                height=320,
                xaxis=dict(title="Probability", gridcolor=GRID_COL,
                           tickformat=".0%", zeroline=False),
                yaxis=dict(title="Count", gridcolor=GRID_COL, zeroline=False),
                showlegend=False,
            )
            st.plotly_chart(fig_hist, use_container_width=True,
                            config={"displayModeBar": False})

    # ── Window-level table ────────────────────────────────────────────────────
    with st.expander("Window-level results table"):
        import pandas as pd
        df = pd.DataFrame([{
            "Window": r["window"],
            "Probability": f"{r['prob']:.3f}",
            "Alert": "YES" if r["alert"] else "no",
            "Risk level": ("Critical" if r["prob"] > 0.75 else
                           "Elevated"  if r["prob"] > 0.45 else "Low"),
        } for r in results])
        st.dataframe(df, use_container_width=True, hide_index=True)

else:
    # ── Empty state ───────────────────────────────────────────────────────────
    st.markdown("""
    <div style='text-align:center; padding: 80px 20px; color: #8b9cc8;'>
      <div style='font-family: Space Mono, monospace; font-size: 3rem;
                  color: rgba(0,212,170,0.15); margin-bottom: 20px;'>⬡</div>
      <div style='font-family: Space Mono, monospace; font-size: 1rem;
                  color: #4a5578; letter-spacing: 0.1em;'>
        AWAITING INPUT
      </div>
      <div style='font-size: 0.85rem; margin-top: 10px; color: #374151;'>
        Upload an EEG file or enable demo mode, then click Run analysis
      </div>
    </div>
    """, unsafe_allow_html=True)
