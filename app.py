"""
Neural Guard — Pre-ictal EEG Seizure Prediction Dashboard
Neuroscience-grade UI with bioluminescent neural aesthetic
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from itertools import combinations
import time
import torch
from pathlib import Path

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Neural Guard · EEG Seizure Prediction",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS — Deep bioluminescent neural theme ────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@300;400;500;600;700&family=Share+Tech+Mono&family=Exo+2:wght@200;300;400;600&display=swap');

:root {
    --void:      #03060f;
    --deep:      #070d1c;
    --surface:   #0c1526;
    --raised:    #111e33;
    --border:    rgba(0,200,255,0.10);
    --border-hi: rgba(0,200,255,0.28);
    --neuron:    #00c8ff;
    --synapse:   #00ffcc;
    --spike:     #ff4d6d;
    --inhibit:   #a855f7;
    --muted:     #4a6080;
    --text:      #c8deff;
    --text-dim:  #5a7090;
}

html, body,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
.main .block-container {
    background-color: var(--void) !important;
    font-family: 'Exo 2', sans-serif !important;
    color: var(--text) !important;
}

[data-testid="stAppViewContainer"]::before {
    content:'';
    position:fixed; inset:0;
    background:
        radial-gradient(ellipse 800px 600px at 18% 28%, rgba(0,200,255,0.03) 0%, transparent 70%),
        radial-gradient(ellipse 600px 800px at 82% 72%, rgba(168,85,247,0.03) 0%, transparent 70%);
    pointer-events:none; z-index:0;
}

[data-testid="stSidebar"] {
    background: var(--deep) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text) !important; }

h1,h2,h3 {
    font-family: 'Rajdhani', sans-serif !important;
    letter-spacing: 0.06em !important;
    color: var(--text) !important;
}

.ng-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 18px 22px;
    margin-bottom: 10px;
    position: relative;
    overflow: hidden;
    transition: border-color 0.3s, box-shadow 0.3s;
}
.ng-card::before {
    content:'';
    position:absolute; top:0; left:0;
    width:3px; height:100%;
    background: linear-gradient(180deg, var(--neuron), var(--synapse));
    opacity:0.7;
}
.ng-card:hover {
    border-color: var(--border-hi);
    box-shadow: 0 0 24px rgba(0,200,255,0.06);
}
.ng-val {
    font-family: 'Share Tech Mono', monospace;
    font-size: 2rem; line-height:1.1; margin: 6px 0 3px;
}
.ng-label {
    font-family: 'Rajdhani', sans-serif;
    font-size: 0.65rem; letter-spacing:0.18em;
    text-transform: uppercase; color: var(--text-dim);
}
.ng-sub { font-size:0.78rem; color:var(--text-dim); margin-top:3px; }

.ng-section {
    font-family: 'Rajdhani', sans-serif;
    font-size: 0.68rem; letter-spacing:0.22em;
    text-transform: uppercase; color: var(--neuron);
    opacity: 0.7; padding-bottom:8px;
    border-bottom: 1px solid var(--border);
    margin-bottom: 14px; margin-top:4px;
}

.ng-alert {
    border-radius:5px; padding:14px 18px; margin:10px 0;
    font-family:'Share Tech Mono',monospace; font-size:0.8rem;
    display:flex; align-items:center; gap:14px; border-left:3px solid;
}
.ng-alert-crit {
    background:rgba(255,77,109,0.08); border-color:var(--spike); color:#ffb3c1;
    animation: pulse-red 2s ease-in-out infinite;
}
.ng-alert-warn {
    background:rgba(245,158,11,0.08); border-color:#f59e0b; color:#fde68a;
}
.ng-alert-safe {
    background:rgba(0,255,204,0.06); border-color:var(--synapse); color:#6ee7d4;
}
@keyframes pulse-red {
    0%,100%{box-shadow:0 0 0 rgba(255,77,109,0);}
    50%{box-shadow:0 0 18px rgba(255,77,109,0.15);}
}

.ng-pill {
    display:inline-block; padding:2px 10px; border-radius:3px;
    font-family:'Share Tech Mono',monospace; font-size:0.68rem;
    letter-spacing:0.05em; border:1px solid;
}
.ng-pill-cyan   {color:var(--neuron); border-color:rgba(0,200,255,0.3);  background:rgba(0,200,255,0.07);}
.ng-pill-green  {color:var(--synapse);border-color:rgba(0,255,204,0.3);  background:rgba(0,255,204,0.07);}
.ng-pill-red    {color:var(--spike);  border-color:rgba(255,77,109,0.3); background:rgba(255,77,109,0.07);}
.ng-pill-purple {color:#c084fc;       border-color:rgba(168,85,247,0.3); background:rgba(168,85,247,0.07);}

[data-testid="stFileUploader"] {
    background:var(--raised) !important;
    border:1px dashed rgba(0,200,255,0.25) !important;
    border-radius:6px !important;
}
.stButton > button {
    background:transparent !important; color:var(--neuron) !important;
    border:1px solid var(--neuron) !important; border-radius:4px !important;
    font-family:'Rajdhani',sans-serif !important; font-size:0.9rem !important;
    font-weight:600 !important; letter-spacing:0.12em !important;
    transition:all 0.2s !important;
}
.stButton > button:hover {
    background:rgba(0,200,255,0.10) !important;
    box-shadow:0 0 16px rgba(0,200,255,0.2) !important;
}
</style>
""", unsafe_allow_html=True)


# ── Plotly shared theme ────────────────────────────────────────────────────────
BG      = "#03060f"
SURFACE = "#0c1526"
GRID    = "rgba(0,200,255,0.05)"
CYAN    = "#00c8ff"
GREEN   = "#00ffcc"
RED     = "#ff4d6d"
PURPLE  = "#a855f7"
MUTED   = "#4a6080"
FONT    = "Exo 2, sans-serif"
MONO    = "Share Tech Mono, monospace"

BASE = dict(
    plot_bgcolor=SURFACE, paper_bgcolor=BG,
    font=dict(family=FONT, color=MUTED),
    margin=dict(l=12, r=12, t=16, b=12),
)


# ── Model loader ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_model(path):
    try:
        from model import SeizureTGNN
        m = SeizureTGNN(n_channels=23, n_samples=7680)
        m.load_state_dict(torch.load(path, map_location="cpu"))
        m.eval()
        return m, None
    except FileNotFoundError:
        return None, "not_found"
    except Exception as e:
        return None, str(e)


# ── Prediction engine ──────────────────────────────────────────────────────────
def run_prediction(edf_path, model, threshold, n_windows=30):
    if model is not None and edf_path and edf_path != "demo":
        try:
            from preprocess import preprocess_edf
            from graph_builder import epoch_to_graph
            from torch_geometric.data import Batch
            epochs = preprocess_edf(edf_path)[:n_windows]
            out = []
            for i, ep in enumerate(epochs):
                g = epoch_to_graph(ep, 0)
                b = Batch.from_data_list([g])
                with torch.no_grad():
                    p = float(torch.softmax(model(b), 1)[0, 1])
                out.append({"window": i, "prob": p, "alert": p >= threshold, "epoch": ep})
            return out, None
        except Exception as e:
            return None, str(e)

    np.random.seed(7)
    out = []
    for i in range(n_windows):
        t = i / n_windows
        if t < 0.50:
            base = 0.07 + np.random.normal(0, 0.03)
        elif t < 0.72:
            base = 0.12 + t * 0.30 + np.random.normal(0, 0.05)
        else:
            base = 0.48 + t * 0.55 + np.random.normal(0, 0.04)
        p = float(np.clip(base, 0.01, 0.98))
        out.append({
            "window": i, "prob": p, "alert": p >= threshold,
            "epoch": np.random.randn(23, 7680).astype(np.float32),
        })
    return out, None


# ── Charts ─────────────────────────────────────────────────────────────────────

def prob_timeline(results, threshold):
    ws    = [r["window"] for r in results]
    probs = [r["prob"]   for r in results]
    pt_c  = [RED if p >= threshold else ("#f59e0b" if p >= threshold*0.72 else GREEN)
             for p in probs]
    fig = go.Figure()
    fig.add_hrect(y0=threshold, y1=1.02,
                  fillcolor="rgba(255,77,109,0.04)", line_width=0)
    fig.add_trace(go.Scatter(x=ws, y=probs, fill="tozeroy",
                             fillcolor="rgba(0,200,255,0.04)",
                             line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=ws, y=probs, mode="lines+markers",
                             line=dict(color=CYAN, width=1.8),
                             marker=dict(color=pt_c, size=6, line=dict(width=0)),
                             hovertemplate="Window %{x} — P(pre-ictal): %{y:.2%}<extra></extra>",
                             showlegend=False))
    fig.add_shape(type="line", x0=ws[0], x1=ws[-1],
                  y0=threshold, y1=threshold,
                  line=dict(color="rgba(245,158,11,0.55)", width=1.2, dash="dot"))
    fig.add_annotation(x=ws[-1], y=threshold+0.04,
                       text=f"Threshold {threshold:.0%}",
                       showarrow=False, xanchor="right",
                       font=dict(size=10, color="#fcd34d", family=MONO))
    fig.update_layout(**BASE, height=250,
        xaxis=dict(title="Window", gridcolor=GRID, zeroline=False,
                   title_font_color=MUTED, tickfont_color=MUTED),
        yaxis=dict(title="P(pre-ictal)", gridcolor=GRID, zeroline=False,
                   tickformat=".0%", range=[0,1.05],
                   title_font_color=MUTED, tickfont_color=MUTED))
    return fig


def gauge(prob):
    color = RED if prob > 0.75 else ("#f59e0b" if prob > 0.45 else GREEN)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(prob * 100, 1),
        number=dict(suffix="%", font=dict(color=color, family=MONO, size=30)),
        gauge=dict(
            axis=dict(
                range=[0, 100],
                tickwidth=1,
                tickfont=dict(color=MUTED, size=9, family=MONO),
                tickcolor=MUTED,      # named color — fixes Plotly v6 validation error
                tickvals=[0, 25, 50, 75, 100],
            ),
            bar=dict(color=color, thickness=0.22),
            bgcolor=SURFACE,
            borderwidth=1,
            bordercolor="rgba(0,200,255,0.12)",
            steps=[
                dict(range=[0,  45], color="rgba(0,255,204,0.07)"),
                dict(range=[45, 75], color="rgba(245,158,11,0.07)"),
                dict(range=[75,100], color="rgba(255,77,109,0.09)"),
            ],
            threshold=dict(
                line=dict(color="rgba(255,255,255,0.25)", width=2),
                thickness=0.78, value=75,
            ),
        ),
    ))
    fig.update_layout(
        plot_bgcolor=SURFACE, paper_bgcolor=BG,
        font=dict(color=MUTED),
        margin=dict(l=16, r=16, t=28, b=8),
        height=195,
    )
    return fig


def eeg_wave(epoch, n_ch=8):
    names = ["FP1","FP2","F7","F3","FZ","F4","F8","T7"][:n_ch]
    t     = np.linspace(0, 30, epoch.shape[1])
    pal   = [CYAN, GREEN, "#7dd3fc", "#86efac",
             "#c084fc", "#fb923c", "#38bdf8", "#4ade80"]
    fig   = go.Figure()
    gap   = 3.2
    for i, name in enumerate(names):
        fig.add_trace(go.Scatter(
            x=t, y=epoch[i] + i * gap, mode="lines",
            line=dict(width=0.7, color=pal[i % len(pal)]),
            hoverinfo="skip", showlegend=False,
        ))
        fig.add_annotation(x=-0.8, y=i*gap, text=name, showarrow=False,
                           font=dict(size=9, color=MUTED, family=MONO), xanchor="right")
    fig.update_layout(**BASE, height=290,
        xaxis=dict(title="Time (s)", gridcolor=GRID, zeroline=False,
                   title_font_color=MUTED, tickfont_color=MUTED),
        yaxis=dict(showticklabels=False, gridcolor=GRID, zeroline=False),
        showlegend=False)
    return fig


def brain_map(epoch):
    POS = {
        "FP1":(-0.30,0.90),"FP2":(0.30,0.90),
        "F7":(-0.70,0.50),"F3":(-0.40,0.62),"FZ":(0.00,0.70),
        "F4":(0.40,0.62),"F8":(0.70,0.50),
        "T7":(-1.00,0.00),"C3":(-0.60,0.00),"CZ":(0.00,0.00),
        "C4":(0.60,0.00),"T8":(1.00,0.00),
        "P7":(-0.70,-0.50),"P3":(-0.40,-0.62),"PZ":(0.00,-0.70),
        "P4":(0.40,-0.62),"P8":(0.70,-0.50),
        "O1":(-0.30,-0.90),"O2":(0.30,-0.90),
        "F1":(-0.20,0.65),"F2":(0.20,0.65),
        "FC1":(-0.35,0.35),"FC2":(0.35,0.35),
    }
    names = list(POS.keys())[:23]
    xs = [POS[n][0] for n in names]
    ys = [POS[n][1] for n in names]
    power = np.mean(epoch[:23]**2, axis=1)
    pn    = (power - power.min()) / (power.max() - power.min() + 1e-8)

    fig = go.Figure()
    th  = np.linspace(0, 2*np.pi, 200)
    fig.add_trace(go.Scatter(x=np.cos(th), y=np.sin(th), mode="lines",
                             line=dict(color="rgba(0,200,255,0.18)", width=1.5),
                             hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=[-.08,0,.08], y=[.98,1.12,.98], mode="lines",
                             line=dict(color="rgba(0,200,255,0.18)", width=1.5),
                             hoverinfo="skip", showlegend=False))

    for i, j in combinations(range(len(names)), 2):
        c = float(np.abs(np.corrcoef(epoch[i], epoch[j])[0, 1]))
        if c > 0.40:
            fig.add_trace(go.Scatter(
                x=[xs[i], xs[j], None], y=[ys[i], ys[j], None], mode="lines",
                line=dict(width=c*2.2, color=f"rgba(168,85,247,{c*0.45:.2f})"),
                hoverinfo="skip", showlegend=False))

    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="markers+text",
        text=names, textposition="top center",
        textfont=dict(size=8, color=MUTED, family=MONO),
        marker=dict(
            size=[10 + pn[i]*14 for i in range(len(names))],
            color=pn,
            colorscale=[[0,"#0c1526"],[0.4,PURPLE],[0.75,CYAN],[1.0,GREEN]],
            showscale=True,
            colorbar=dict(
                title=dict(text="Power", font=dict(color=MUTED, size=9)),
                thickness=8, tickfont=dict(color=MUTED, size=8), x=1.02,
            ),
            line=dict(color="rgba(0,200,255,0.3)", width=1),
        ),
        hovertemplate="%{text}<br>Power: %{marker.color:.3f}<extra></extra>",
        showlegend=False,
    ))
    fig.update_layout(**BASE, height=330,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-1.3,1.35]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                   range=[-1.2,1.25], scaleanchor="x"))
    return fig


def prob_histogram(probs, threshold):
    fig = go.Figure(go.Histogram(
        x=probs, nbinsx=14,
        marker_color=PURPLE,
        marker_line_color=BG, marker_line_width=1.2,
    ))
    fig.add_vline(x=threshold, line_dash="dot", line_color="#f59e0b", line_width=1.5,
                  annotation_text="threshold",
                  annotation_font=dict(color="#fcd34d", size=9, family=MONO),
                  annotation_position="top right")
    fig.update_layout(**BASE, height=200,
        xaxis=dict(title="P(pre-ictal)", gridcolor=GRID, zeroline=False,
                   tickformat=".0%", title_font_color=MUTED, tickfont_color=MUTED),
        yaxis=dict(title="Count", gridcolor=GRID, zeroline=False,
                   title_font_color=MUTED, tickfont_color=MUTED),
        showlegend=False)
    return fig


def channel_power_bar(epoch):
    names = ["FP1","FP2","F7","F3","FZ","F4","F8","T7",
             "C3","CZ","C4","T8","P7","P3","PZ","P4",
             "P8","O1","O2","F1","F2","FC1","FC2"]
    power  = np.mean(epoch[:23]**2, axis=1)
    pn     = power / (power.max() + 1e-8)
    colors = [RED if p > 0.75 else (CYAN if p > 0.4 else MUTED) for p in pn]
    fig = go.Figure(go.Bar(
        x=names[:len(power)], y=power,
        marker_color=colors, marker_line_width=0,
    ))
    fig.update_layout(**BASE, height=200,
        xaxis=dict(gridcolor=GRID, zeroline=False,
                   tickfont=dict(size=8, color=MUTED, family=MONO)),
        yaxis=dict(title="Mean power", gridcolor=GRID, zeroline=False,
                   title_font_color=MUTED, tickfont_color=MUTED),
        showlegend=False)
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='padding:10px 0 24px;'>
      <div style='font-family:Rajdhani,sans-serif;font-size:1.3rem;
                  font-weight:700;color:#00c8ff;letter-spacing:0.12em;'>
        NEURAL GUARD</div>
      <div style='font-family:Share Tech Mono,monospace;font-size:0.65rem;
                  color:#4a6080;letter-spacing:0.18em;margin-top:3px;'>
        PRE-ICTAL EEG ANALYSIS  v2.0</div>
      <div style='width:40px;height:1px;background:#00c8ff;
                  opacity:0.4;margin-top:10px;'></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="ng-section">EEG Input</div>', unsafe_allow_html=True)
    uploaded  = st.file_uploader("Upload .edf file", type=["edf"])
    demo_mode = st.checkbox("Demo mode (synthetic data)", value=True)

    st.markdown('<div class="ng-section">Model</div>', unsafe_allow_html=True)
    model_path = st.text_input("Checkpoint path", value="best_model.pt")

    st.markdown('<div class="ng-section">Detection Parameters</div>', unsafe_allow_html=True)
    threshold = st.slider("Alert threshold",  0.40, 0.95, 0.75, 0.05)
    n_windows = st.slider("Analysis windows", 10,   60,   30,   5)
    horizon   = st.slider("Sustained horizon", 2,    8,    5)

    st.markdown('<div class="ng-section">Display</div>', unsafe_allow_html=True)
    show_eeg   = st.checkbox("EEG waveform",            value=True)
    show_brain = st.checkbox("Brain connectivity map",  value=True)
    show_power = st.checkbox("Channel power spectrum",  value=True)

    st.markdown("""
    <div style='margin-top:28px;padding:14px;background:#070d1c;
                border:1px solid rgba(0,200,255,0.08);border-radius:5px;'>
      <div style='font-family:Rajdhani,sans-serif;font-size:0.62rem;
                  letter-spacing:0.15em;color:#4a6080;text-transform:uppercase;
                  margin-bottom:8px;'>System Info</div>
      <div style='font-family:Share Tech Mono,monospace;font-size:0.68rem;
                  color:#4a6080;line-height:2;'>
        Dataset · CHB-MIT<br>Arch · STGCN<br>
        Edges · PLV<br>Loss · Focal γ=2<br>Adapt · Few-shot
      </div>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════════════════════
model, merr = load_model(model_path)

st.markdown("""
<div style='padding:6px 0 18px;border-bottom:1px solid rgba(0,200,255,0.08);
            margin-bottom:18px;'>
  <span style='font-family:Rajdhani,sans-serif;font-size:1.8rem;
               font-weight:700;color:#c8deff;letter-spacing:0.04em;'>
    Neural Guard</span>
  <span style='font-family:Share Tech Mono,monospace;font-size:0.75rem;
               color:#00c8ff;margin-left:14px;letter-spacing:0.08em;'>
    PRE-ICTAL SEIZURE PREDICTION</span>
</div>
""", unsafe_allow_html=True)

pc1,pc2,pc3,pc4,pc5 = st.columns(5)
with pc1:
    lbl = "Model active" if model else "Demo mode"
    cls = "ng-pill-cyan" if model else "ng-pill-purple"
    st.markdown(f'<span class="ng-pill {cls}">{lbl}</span>', unsafe_allow_html=True)
with pc2:
    st.markdown(f'<span class="ng-pill ng-pill-cyan">THR {threshold:.0%}</span>',
                unsafe_allow_html=True)
with pc3:
    st.markdown(f'<span class="ng-pill ng-pill-purple">WIN {n_windows}</span>',
                unsafe_allow_html=True)
with pc4:
    st.markdown(f'<span class="ng-pill ng-pill-purple">HRZ {horizon}</span>',
                unsafe_allow_html=True)
with pc5:
    st.markdown('<span class="ng-pill ng-pill-green">EEG 256Hz</span>',
                unsafe_allow_html=True)

st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

btn_col, _ = st.columns([1, 4])
with btn_col:
    run = st.button("⬡  RUN NEURAL SCAN", use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# RESULTS
# ═══════════════════════════════════════════════════════════════════════════════
if run or "ng_results" in st.session_state:

    if run:
        edf = None
        if uploaded and not demo_mode:
            p = Path("_tmp.edf")
            p.write_bytes(uploaded.read())
            edf = str(p)

        steps = [
            (18,  "Initialising neural scan..."),
            (38,  "Bandpass filtering 0.5–40 Hz..."),
            (58,  "Computing PLV brain graph..."),
            (78,  "Running Temporal GNN inference..."),
            (95,  "Aggregating window predictions..."),
            (100, "Scan complete."),
        ]
        prog = st.progress(0, text=steps[0][1])
        for pct, msg in steps:
            time.sleep(0.18)
            prog.progress(pct, text=msg)
        prog.empty()

        results, err = run_prediction(edf or "demo", model, threshold, n_windows)
        if err:
            st.error(f"Error: {err}")
            st.stop()
        st.session_state["ng_results"] = results

    results = st.session_state.get("ng_results", [])
    if not results:
        st.stop()

    probs     = [r["prob"]  for r in results]
    alerts    = [r["alert"] for r in results]
    max_p     = max(probs)
    mean_p    = float(np.mean(probs))
    n_alert   = sum(alerts)
    sustained = any(
        all(alerts[i:i+horizon])
        for i in range(max(1, len(alerts) - horizon + 1))
    )

    # Alert banner
    if sustained:
        st.markdown(f"""
        <div class="ng-alert ng-alert-crit">
          <span style='font-size:1.1rem'>⚠</span>
          SEIZURE ALERT — Sustained pre-ictal activity across {horizon}+ windows.
          Peak: {max_p:.1%} &nbsp;|&nbsp; Immediate clinical attention advised.
        </div>""", unsafe_allow_html=True)
    elif n_alert > 0:
        st.markdown(f"""
        <div class="ng-alert ng-alert-warn">
          <span style='font-size:1.1rem'>△</span>
          ELEVATED RISK — {n_alert} window(s) exceeded threshold.
          Peak: {max_p:.1%} &nbsp;|&nbsp; Monitoring for sustained pattern.
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="ng-alert ng-alert-safe">
          <span style='font-size:1.1rem'>✓</span>
          INTERICTAL — No pre-ictal activity detected.
          Max probability: {max_p:.1%} &nbsp;|&nbsp; All windows normal.
        </div>""", unsafe_allow_html=True)

    # Metric cards
    m1,m2,m3,m4 = st.columns(4)
    pk_c  = RED if max_p  > threshold else GREEN
    al_c  = RED if n_alert > 0        else GREEN
    st_t  = "ALERT" if sustained else ("WARNING" if n_alert > 0 else "CLEAR")
    st_c  = RED if sustained else ("#f59e0b" if n_alert > 0 else GREEN)

    with m1:
        st.markdown(f"""
        <div class="ng-card">
          <div class="ng-label">Peak probability</div>
          <div class="ng-val" style="color:{pk_c}">{max_p:.1%}</div>
          <div class="ng-sub">Highest single window</div>
        </div>""", unsafe_allow_html=True)
    with m2:
        st.markdown(f"""
        <div class="ng-card">
          <div class="ng-label">Session mean</div>
          <div class="ng-val" style="color:{CYAN}">{mean_p:.1%}</div>
          <div class="ng-sub">Across {len(results)} windows</div>
        </div>""", unsafe_allow_html=True)
    with m3:
        st.markdown(f"""
        <div class="ng-card">
          <div class="ng-label">Alert windows</div>
          <div class="ng-val" style="color:{al_c}">{n_alert} / {len(results)}</div>
          <div class="ng-sub">Above {threshold:.0%} threshold</div>
        </div>""", unsafe_allow_html=True)
    with m4:
        st.markdown(f"""
        <div class="ng-card">
          <div class="ng-label">Clinical status</div>
          <div class="ng-val" style="color:{st_c};font-size:1.5rem;">{st_t}</div>
          <div class="ng-sub">Horizon: {horizon} windows</div>
        </div>""", unsafe_allow_html=True)

    # Timeline + gauge
    tl, gc = st.columns([3, 1])
    with tl:
        st.markdown('<div class="ng-section">Pre-ictal probability timeline</div>',
                    unsafe_allow_html=True)
        st.plotly_chart(prob_timeline(results, threshold),
                        use_container_width=True, config={"displayModeBar": False})
    with gc:
        st.markdown('<div class="ng-section">Latest window</div>',
                    unsafe_allow_html=True)
        st.plotly_chart(gauge(results[-1]["prob"]),
                        use_container_width=True, config={"displayModeBar": False})
        lp   = results[-1]["prob"]
        risk = "CRITICAL" if lp > 0.75 else ("ELEVATED" if lp > 0.45 else "NORMAL")
        rc   = RED if lp > 0.75 else ("#f59e0b" if lp > 0.45 else GREEN)
        st.markdown(f"""
        <div style='text-align:center;font-family:Share Tech Mono,monospace;
                    font-size:0.72rem;color:{rc};margin-top:-6px;
                    letter-spacing:0.1em;'>{risk}</div>""", unsafe_allow_html=True)

    if show_eeg:
        st.markdown('<div class="ng-section">Raw EEG — last window (8 channels)</div>',
                    unsafe_allow_html=True)
        st.plotly_chart(eeg_wave(results[-1]["epoch"]),
                        use_container_width=True, config={"displayModeBar": False})

    if show_brain:
        b1, b2 = st.columns([3, 2])
        with b1:
            st.markdown('<div class="ng-section">Brain connectivity — functional graph</div>',
                        unsafe_allow_html=True)
            st.plotly_chart(brain_map(results[-1]["epoch"]),
                            use_container_width=True, config={"displayModeBar": False})
            st.markdown("""
            <div style='font-family:Share Tech Mono,monospace;font-size:0.68rem;
                        color:#4a6080;margin-top:-8px;'>
              Node size ∝ channel power &nbsp;·&nbsp;
              Edge weight ∝ correlation &nbsp;·&nbsp;
              Color: low→purple→cyan→green
            </div>""", unsafe_allow_html=True)
        with b2:
            st.markdown('<div class="ng-section">Probability distribution</div>',
                        unsafe_allow_html=True)
            st.plotly_chart(prob_histogram(probs, threshold),
                            use_container_width=True, config={"displayModeBar": False})
            if show_power:
                st.markdown('<div class="ng-section">Channel power spectrum</div>',
                            unsafe_allow_html=True)
                st.plotly_chart(channel_power_bar(results[-1]["epoch"]),
                                use_container_width=True, config={"displayModeBar": False})

    with st.expander("Full window-level results"):
        import pandas as pd
        df = pd.DataFrame([{
            "Window":       r["window"],
            "P(pre-ictal)": f"{r['prob']:.4f}",
            "Alert":        "YES" if r["alert"] else "—",
            "Risk":         ("Critical" if r["prob"] > 0.75 else
                             "Elevated"  if r["prob"] > 0.45 else "Normal"),
        } for r in results])
        st.dataframe(df, use_container_width=True, hide_index=True)

else:
    st.markdown("""
    <div style='text-align:center;padding:90px 0 70px;'>
      <div style='font-family:Rajdhani,sans-serif;font-size:4rem;font-weight:300;
                  color:rgba(0,200,255,0.12);letter-spacing:0.3em;'>⬡ ⬡ ⬡</div>
      <div style='font-family:Share Tech Mono,monospace;font-size:0.9rem;
                  color:#1e3050;letter-spacing:0.2em;margin-top:20px;'>
        AWAITING NEURAL SCAN</div>
      <div style='font-size:0.82rem;color:#1a2a40;margin-top:10px;
                  font-family:Exo 2,sans-serif;'>
        Enable demo mode and click RUN NEURAL SCAN to begin</div>
    </div>
    """, unsafe_allow_html=True)
