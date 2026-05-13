import streamlit as st
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
import plotly.graph_objects as go
import os
from datetime import datetime

# ── Page config ───────────────────────────────────────────
st.set_page_config(
    page_title="Bankruptcy Probability Estimator",
    layout="wide"
)

# ── Custom CSS ────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background-color: #f8fafc;
}

/* ── Hide default Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container {
    padding-top: 1.5rem !important;
    padding-bottom: 3rem !important;
    max-width: 1100px;
}

/* ── Top navbar ── */
.navbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: #0f172a;
    padding: 0.85rem 2rem;
    border-radius: 12px;
    margin-bottom: 2rem;
}
.navbar-brand {
    display: flex;
    align-items: center;
    gap: 0.7rem;
}
.navbar-brand .dot {
    width: 10px;
    height: 10px;
    border-radius: 50%;
    background: #3b82f6;
    display: inline-block;
}
.navbar-brand span {
    color: white;
    font-size: 1rem;
    font-weight: 600;
    letter-spacing: 0.02em;
}
.navbar-tag {
    background: rgba(59,130,246,0.15);
    color: #93c5fd;
    font-size: 0.72rem;
    font-weight: 500;
    padding: 0.25rem 0.7rem;
    border-radius: 20px;
    border: 1px solid rgba(59,130,246,0.3);
    letter-spacing: 0.04em;
    text-transform: uppercase;
}

/* ── Hero section ── */
.hero {
    background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 50%, #1e40af 100%);
    border-radius: 16px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero::after {
    content: '';
    position: absolute;
    top: -40px; right: -40px;
    width: 200px; height: 200px;
    background: rgba(59,130,246,0.12);
    border-radius: 50%;
}
.hero h1 {
    color: #ffffff;
    font-size: 1.85rem;
    font-weight: 700;
    margin: 0 0 0.5rem 0;
    letter-spacing: -0.02em;
}
.hero p {
    color: #94a3b8;
    font-size: 0.95rem;
    margin: 0;
    max-width: 600px;
    line-height: 1.6;
}
.hero-badge {
    display: inline-block;
    background: rgba(59,130,246,0.2);
    color: #60a5fa;
    font-size: 0.72rem;
    font-weight: 600;
    padding: 0.2rem 0.65rem;
    border-radius: 4px;
    margin-bottom: 0.8rem;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    border: 1px solid rgba(59,130,246,0.3);
}

/* ── Step cards ── */
.step-wrap {
    display: flex;
    gap: 0.8rem;
    align-items: flex-start;
    margin-bottom: 1.2rem;
}
.step-num {
    min-width: 34px;
    height: 34px;
    background: #1d4ed8;
    color: white;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.85rem;
    font-weight: 700;
    margin-top: 0.1rem;
    flex-shrink: 0;
}
.step-body {
    background: white;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    border: 1px solid #e2e8f0;
    flex: 1;
    box-shadow: 0 1px 4px rgba(0,0,0,0.04);
}
.step-body h4 {
    color: #0f172a;
    font-size: 0.95rem;
    font-weight: 600;
    margin: 0 0 0.3rem 0;
}
.step-body p {
    color: #64748b;
    font-size: 0.85rem;
    margin: 0;
    line-height: 1.55;
}

/* ── Results card ── */
.results-card {
    background: white;
    border-radius: 14px;
    padding: 1.8rem 2rem;
    border: 1px solid #e2e8f0;
    margin-bottom: 1.5rem;
    box-shadow: 0 2px 10px rgba(0,0,0,0.05);
}
.results-card h3 {
    color: #0f172a;
    font-size: 1rem;
    font-weight: 600;
    margin: 0 0 1rem 0;
    padding-bottom: 0.8rem;
    border-bottom: 1px solid #f1f5f9;
}

/* ── KPI tiles ── */
.kpi-grid {
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    gap: 1rem;
    margin-bottom: 1rem;
}
.kpi-tile {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    text-align: center;
}
.kpi-tile .kpi-label {
    font-size: 0.7rem;
    font-weight: 600;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    margin-bottom: 0.4rem;
}
.kpi-tile .kpi-value {
    font-size: 1.65rem;
    font-weight: 700;
    line-height: 1;
}
.kpi-tile.high  { border-top: 3px solid #ef4444; }
.kpi-tile.high .kpi-value  { color: #dc2626; }
.kpi-tile.med   { border-top: 3px solid #f59e0b; }
.kpi-tile.med  .kpi-value  { color: #d97706; }
.kpi-tile.low   { border-top: 3px solid #10b981; }
.kpi-tile.low  .kpi-value  { color: #059669; }

/* ── Alert banners ── */
.alert {
    border-radius: 10px;
    padding: 1rem 1.3rem;
    font-size: 0.87rem;
    line-height: 1.5;
    margin-top: 1rem;
    display: flex;
    gap: 0.7rem;
    align-items: flex-start;
}
.alert-icon { font-size: 1.1rem; flex-shrink: 0; margin-top: 0.05rem; }
.alert.high { background: #fef2f2; border: 1px solid #fecaca; color: #7f1d1d; }
.alert.med  { background: #fffbeb; border: 1px solid #fde68a; color: #78350f; }
.alert.low  { background: #f0fdf4; border: 1px solid #bbf7d0; color: #14532d; }
.alert strong { display: block; margin-bottom: 0.2rem; font-weight: 600; font-size: 0.9rem; }

/* ── Section label ── */
.section-label {
    font-size: 0.7rem;
    font-weight: 700;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin: 1.8rem 0 0.7rem 0;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #0f172a !important;
    border-right: 1px solid #1e293b;
}
[data-testid="stSidebar"] .stMarkdown,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] li,
[data-testid="stSidebar"] span {
    color: #94a3b8 !important;
    font-size: 0.83rem !important;
}
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] strong {
    color: #e2e8f0 !important;
}
[data-testid="stSidebar"] hr {
    border-color: #1e293b !important;
    margin: 0.8rem 0 !important;
}
.sidebar-stat {
    background: #1e293b;
    border-radius: 8px;
    padding: 0.65rem 0.9rem;
    margin-bottom: 0.5rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.sidebar-stat .s-label { color: #64748b; font-size: 0.78rem; }
.sidebar-stat .s-value { color: #60a5fa; font-size: 0.85rem; font-weight: 600; }

/* ── Download button ── */
.stDownloadButton > button {
    background: #1d4ed8 !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    padding: 0.55rem 1.4rem !important;
    transition: background 0.2s !important;
}
.stDownloadButton > button:hover {
    background: #2563eb !important;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: white;
    border-radius: 10px;
    border: 2px dashed #cbd5e1 !important;
    padding: 0.3rem 0.5rem;
}

/* ── Expander ── */
[data-testid="stExpander"] {
    background: white;
    border-radius: 10px;
    border: 1px solid #e2e8f0 !important;
}

/* ── Dataframe ── */
[data-testid="stDataFrame"] {
    border-radius: 10px;
    overflow: hidden;
    border: 1px solid #e2e8f0;
}

/* ── Success/warning/error override ── */
.stAlert { border-radius: 10px !important; }

/* ── Feedback card ── */
.feedback-card {
    background: white;
    border-radius: 14px;
    padding: 2rem 2.2rem;
    border: 1px solid #e2e8f0;
    margin-top: 2rem;
    box-shadow: 0 2px 10px rgba(0,0,0,0.05);
}
.feedback-card .fb-header {
    display: flex;
    align-items: center;
    gap: 0.7rem;
    margin-bottom: 0.4rem;
}
.feedback-card .fb-icon {
    width: 36px;
    height: 36px;
    background: #eff6ff;
    border-radius: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1rem;
}
.feedback-card h3 {
    color: #0f172a;
    font-size: 1rem;
    font-weight: 600;
    margin: 0;
}
.feedback-card .fb-sub {
    color: #64748b;
    font-size: 0.83rem;
    margin: 0 0 1.5rem 0;
    line-height: 1.5;
}
.fb-divider {
    border: none;
    border-top: 1px solid #f1f5f9;
    margin: 0.8rem 0 1.4rem 0;
}
.fb-submitted {
    background: #f0fdf4;
    border: 1px solid #bbf7d0;
    border-radius: 10px;
    padding: 1rem 1.3rem;
    color: #14532d;
    font-size: 0.87rem;
    display: flex;
    align-items: center;
    gap: 0.6rem;
}
.fb-submitted strong { font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ── Load model ────────────────────────────────────────────
@st.cache_resource
def load_model():
    m = CatBoostClassifier()
    m.load_model("model/catboost_bankruptcy.cbm")
    return m

model = load_model()

# ── Feature mapping ───────────────────────────────────────
FEATURES = ['X1','X2','X3','X4','X5','X6','X7','X8',
            'X9','X10','X11','X12','X13','X14','X15',
            'X16','X17','X18','Division']

DISPLAY_NAMES = [
    'Current Assets', 'Cost of Goods Sold', 'Depreciation and Amortization',
    'EBITDA', 'Inventory', 'Net Income', 'Total Receivables', 'Market Value',
    'Net Sales', 'Total Assets', 'Total Long-term Debt', 'EBIT', 'Gross Profit',
    'Retained Earnings', 'Total Revenue',
    'Total Liabilities', 'Total Operating Expenses', 'Division'
]

USER_FEATURES = ['X1','X2','X3','X4','X5','X6','X7','X8',
                 'X9','X10','X11','X12','X13','X15',
                 'X16','X17','X18','Division']

NAME_TO_FEATURE = dict(zip(DISPLAY_NAMES, USER_FEATURES))
FEATURE_TO_NAME = dict(zip(USER_FEATURES, DISPLAY_NAMES))

# ── Sidebar ───────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Bankruptcy Probability Estimator")
    st.markdown("---")
    st.markdown("**About**")
    st.markdown(
        "CatBoost gradient boosting model trained on 78,682 records "
        "of US public companies (NYSE & NASDAQ), fiscal years 1999–2018."
    )
    st.markdown("---")
    st.markdown("**Model Performance**")
    st.markdown("""
<div class="sidebar-stat"><span class="s-label">AUC Score</span><span class="s-value">0.8281</span></div>
<div class="sidebar-stat"><span class="s-label">Recall (Failed)</span><span class="s-value">70.4%</span></div>
<div class="sidebar-stat"><span class="s-label">Decision Threshold</span><span class="s-value">0.40</span></div>
""", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("**Risk Classification**")
    st.markdown("""
<div class="sidebar-stat"><span class="s-label">Low Risk</span><span class="s-value">0 – 20%</span></div>
<div class="sidebar-stat"><span class="s-label">Medium Risk</span><span class="s-value">20 – 40%</span></div>
<div class="sidebar-stat"><span class="s-label">High Risk</span><span class="s-value">> 40%</span></div>
""", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown(
        "<p style='font-size:0.75rem;color:#475569;line-height:1.6;'>"
        "Romanian-American University<br>"
        "Computer Science for Economics<br>"
        "Bachelor Thesis — 2026</p>",
        unsafe_allow_html=True
    )

# ── Navbar ────────────────────────────────────────────────
st.markdown("""
<div class="navbar">
    <div class="navbar-brand">
        <span class="dot"></span>
        <span>Bankruptcy Probability Estimator</span>
    </div>
    <span class="navbar-tag">ML-Powered</span>
</div>
""", unsafe_allow_html=True)

# ── Hero ──────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-badge">Financial Distress Prediction</div>
    <h1>Bankruptcy Probability Estimator</h1>
    <p>Upload a company's annual financial data to receive an instant,
    machine-learning-powered risk assessment based on 18 key financial indicators.</p>
</div>
""", unsafe_allow_html=True)

# ── Step 1 ────────────────────────────────────────────────
st.markdown("""
<div class="step-wrap">
    <div class="step-num">1</div>
    <div class="step-body">
        <h4>Download the Input Template</h4>
        <p>Download the CSV template and fill in the company's financial figures from its annual report.
        Leave unknown values as 0. The Division field accepts letters A–J (industry sector codes).</p>
    </div>
</div>
""", unsafe_allow_html=True)

template_df = pd.DataFrame(columns=DISPLAY_NAMES)
template_df.loc[0] = [0.0] * 17 + ['D']

st.download_button(
    label="Download CSV Template",
    data=template_df.to_csv(index=False).encode('utf-8'),
    file_name="company_financial_template.csv",
    mime="text/csv"
)

st.markdown("<div style='height:1.2rem'></div>", unsafe_allow_html=True)

# ── Step 2 ────────────────────────────────────────────────
st.markdown("""
<div class="step-wrap">
    <div class="step-num">2</div>
    <div class="step-body">
        <h4>Upload the Completed File</h4>
        <p>Upload the filled template as a CSV or Excel (.xlsx) file.
        You can include multiple companies — one row per company.</p>
    </div>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("", type=["csv", "xlsx"], label_visibility="collapsed")

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        df = df.rename(columns=NAME_TO_FEATURE)

        for f in FEATURES[:-1]:
            if f not in df.columns:
                df[f] = 0.0
            else:
                df[f] = df[f].fillna(0.0)

        if 'Division' not in df.columns:
            df['Division'] = 'D'
        else:
            df['Division'] = df['Division'].fillna('D')

        df['X14'] = df['X17']

        st.success(f"File loaded — {len(df)} company record(s) found.")

        with st.expander("View uploaded data"):
            display_df = df[FEATURES].rename(columns=FEATURE_TO_NAME)
            st.dataframe(display_df, use_container_width=True)

        st.markdown("<div style='height:1.2rem'></div>", unsafe_allow_html=True)

        # ── Step 3 ────────────────────────────────────────
        st.markdown("""
<div class="step-wrap">
    <div class="step-num">3</div>
    <div class="step-body">
        <h4>Prediction Results</h4>
        <p>The model outputs a bankruptcy probability for each company.
        Scores above 40% are classified as <strong>AT RISK</strong>.</p>
    </div>
</div>
""", unsafe_allow_html=True)

        X = df[FEATURES]
        cat_idx = [FEATURES.index('Division')]
        pool = Pool(X, cat_features=cat_idx)

        raw_proba   = model.predict_proba(pool)[:, 1]
        predictions = (raw_proba >= 0.4).astype(int)

        results = pd.DataFrame()
        if 'company_name' in df.columns:
            results['Company'] = df['company_name']
        results['Bankruptcy Probability (%)'] = (raw_proba * 100).round(2)
        results['Prediction'] = ['AT RISK' if p == 1 else 'STABLE' for p in predictions]
        results['Risk Level'] = pd.cut(
            raw_proba, bins=[0, 0.2, 0.4, 1.0], labels=['Low', 'Medium', 'High']
        )

        def highlight_risk(row):
            if row['Risk Level'] == 'High':
                return ['background-color: #fef2f2; color: #991b1b'] * len(row)
            elif row['Risk Level'] == 'Medium':
                return ['background-color: #fffbeb; color: #92400e'] * len(row)
            else:
                return ['background-color: #f0fdf4; color: #14532d'] * len(row)

        st.dataframe(
            results.style.apply(highlight_risk, axis=1),
            use_container_width=True
        )

        # ── Single company detail ─────────────────────────
        if len(df) == 1:
            prob = raw_proba[0] * 100
            risk = str(results['Risk Level'].iloc[0])
            pred = results['Prediction'].iloc[0]
            css_class = {'High': 'high', 'Medium': 'med', 'Low': 'low'}.get(risk, 'low')

            st.markdown('<p class="section-label">Risk Assessment</p>', unsafe_allow_html=True)

            col1, col2 = st.columns([1.15, 1], gap="large")

            with col1:
                needle_color = {'High': '#ef4444', 'Medium': '#f59e0b', 'Low': '#10b981'}.get(risk, '#10b981')
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=round(prob, 2),
                    number={
                        'suffix': '%',
                        'font': {'size': 48, 'color': needle_color, 'family': 'Inter'}
                    },
                    title={
                        'text': "Bankruptcy Probability",
                        'font': {'size': 14, 'color': '#64748b', 'family': 'Inter'}
                    },
                    gauge={
                        'axis': {
                            'range': [0, 100],
                            'tickfont': {'size': 11, 'color': '#94a3b8'},
                            'tickwidth': 1,
                            'tickcolor': '#e2e8f0'
                        },
                        'bar': {'color': needle_color, 'thickness': 0.22},
                        'bgcolor': 'white',
                        'borderwidth': 0,
                        'steps': [
                            {'range': [0,  20],  'color': '#f0fdf4'},
                            {'range': [20, 40],  'color': '#fffbeb'},
                            {'range': [40, 100], 'color': '#fef2f2'},
                        ],
                        'threshold': {
                            'line': {'color': '#94a3b8', 'width': 2},
                            'thickness': 0.7,
                            'value': 40
                        }
                    }
                ))
                fig.update_layout(
                    height=300,
                    margin=dict(l=20, r=20, t=50, b=10),
                    paper_bgcolor='white',
                    plot_bgcolor='white',
                    font={'family': 'Inter'}
                )
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.markdown(f"""
<div style="display:flex;flex-direction:column;gap:0.9rem;padding-top:0.5rem;">
    <div class="kpi-tile {css_class}">
        <div class="kpi-label">Bankruptcy Probability</div>
        <div class="kpi-value">{prob:.2f}%</div>
    </div>
    <div class="kpi-tile {css_class}">
        <div class="kpi-label">Risk Level</div>
        <div class="kpi-value">{risk}</div>
    </div>
    <div class="kpi-tile {css_class}">
        <div class="kpi-label">Prediction</div>
        <div class="kpi-value" style="font-size:1.25rem;">{pred}</div>
    </div>
</div>
""", unsafe_allow_html=True)

            # Alert
            if risk == 'High':
                st.markdown("""
<div class="alert high">
    <span class="alert-icon">&#9888;</span>
    <div><strong>High Bankruptcy Risk Detected</strong>
    This company's financial profile is consistent with firms that have entered financial distress.
    Further investigation is strongly recommended before any investment or credit decision.</div>
</div>""", unsafe_allow_html=True)
            elif risk == 'Medium':
                st.markdown("""
<div class="alert med">
    <span class="alert-icon">&#8505;</span>
    <div><strong>Moderate Bankruptcy Risk</strong>
    Some financial indicators suggest vulnerability. Monitor liquidity ratios,
    debt coverage, and profitability trends closely.</div>
</div>""", unsafe_allow_html=True)
            else:
                st.markdown("""
<div class="alert low">
    <span class="alert-icon">&#10003;</span>
    <div><strong>Low Bankruptcy Risk</strong>
    This company appears financially stable based on the provided indicators.
    No immediate financial distress signals detected.</div>
</div>""", unsafe_allow_html=True)

            # Feature importance
            st.markdown('<p class="section-label">Key Financial Indicators</p>', unsafe_allow_html=True)
            st.markdown(
                "<p style='font-size:0.83rem;color:#64748b;margin-bottom:0.8rem;'>"
                "Financial indicators ranked by their influence on the model's prediction.</p>",
                unsafe_allow_html=True
            )

            importances = model.get_feature_importance()
            numeric_features = FEATURES[:-1]
            fi = pd.DataFrame({
                'Indicator': [FEATURE_TO_NAME.get(f, f) for f in numeric_features],
                'Importance': importances[:len(numeric_features)]
            }).sort_values('Importance', ascending=True).tail(10)

            bar_colors = [
                '#bfdbfe','#bfdbfe','#bfdbfe','#bfdbfe','#bfdbfe',
                '#bfdbfe','#93c5fd','#60a5fa','#3b82f6','#1d4ed8'
            ]

            fig2 = go.Figure(go.Bar(
                x=fi['Importance'],
                y=fi['Indicator'],
                orientation='h',
                marker_color=bar_colors[:len(fi)],
                marker_line_width=0,
                text=fi['Importance'].round(1),
                textposition='outside',
                textfont={'size': 10, 'color': '#64748b', 'family': 'Inter'}
            ))
            fig2.update_layout(
                xaxis_title=None,
                xaxis=dict(
                    showgrid=True, gridcolor='#f1f5f9',
                    showline=False, zeroline=False,
                    tickfont={'size': 10, 'color': '#94a3b8'}
                ),
                yaxis=dict(
                    tickfont={'size': 11, 'color': '#334155', 'family': 'Inter'},
                    showline=False
                ),
                height=380,
                margin=dict(l=0, r=50, t=10, b=20),
                paper_bgcolor='white',
                plot_bgcolor='white',
                font={'family': 'Inter'}
            )
            st.plotly_chart(fig2, use_container_width=True)

        # ── Multi-company view ────────────────────────────
        else:
            n_high  = int((results['Risk Level'] == 'High').sum())
            n_med   = int((results['Risk Level'] == 'Medium').sum())
            n_low   = int((results['Risk Level'] == 'Low').sum())
            n_total = len(results)

            st.markdown('<p class="section-label">Portfolio Summary</p>', unsafe_allow_html=True)

            c1, c2, c3, c4 = st.columns(4)
            tiles = [
                (c1, n_total, "Total Companies", ""),
                (c2, n_high,  "High Risk",        "high"),
                (c3, n_med,   "Medium Risk",       "med"),
                (c4, n_low,   "Low Risk",          "low"),
            ]
            for col, val, label, cls in tiles:
                with col:
                    st.markdown(f"""
<div class="kpi-tile {cls}" style="margin-bottom:0;">
    <div class="kpi-label">{label}</div>
    <div class="kpi-value">{val}</div>
</div>""", unsafe_allow_html=True)

            st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

            fig3 = go.Figure(go.Bar(
                x=['Low Risk', 'Medium Risk', 'High Risk'],
                y=[n_low, n_med, n_high],
                marker_color=['#10b981', '#f59e0b', '#ef4444'],
                marker_line_width=0,
                text=[n_low, n_med, n_high],
                textposition='outside',
                textfont={'size': 12, 'color': '#334155', 'family': 'Inter'}
            ))
            fig3.update_layout(
                yaxis_title="Number of Companies",
                xaxis=dict(showgrid=False, tickfont={'size': 12, 'color': '#334155'}),
                yaxis=dict(showgrid=True, gridcolor='#f1f5f9', tickfont={'size': 11}),
                height=300,
                margin=dict(t=30, b=10, l=10, r=10),
                paper_bgcolor='white',
                plot_bgcolor='white',
                font={'family': 'Inter'}
            )
            st.plotly_chart(fig3, use_container_width=True)

        # ── Feedback section ──────────────────────────────
        st.markdown("""
<div class="feedback-card">
    <div class="fb-header">
        <div class="fb-icon">💬</div>
        <h3>Was this prediction accurate?</h3>
    </div>
    <hr class="fb-divider">
    <p class="fb-sub">Your feedback helps improve the model over time.
    If you know the actual outcome for this company, please let us know.</p>
</div>
""", unsafe_allow_html=True)

        with st.form("feedback_form", clear_on_submit=True):
            fb_col1, fb_col2 = st.columns(2)

            with fb_col1:
                accuracy = st.radio(
                    "How accurate was the prediction?",
                    options=["✅  Correct — matches the real outcome",
                             "❌  Incorrect — real outcome was different",
                             "🤷  Not sure yet"],
                    index=2
                )
                company_label = st.text_input(
                    "Company name or identifier (optional)",
                    placeholder="e.g. Company ABC, Ticker XYZ"
                )

            with fb_col2:
                actual_outcome = st.selectbox(
                    "Actual outcome (if known)",
                    options=["Unknown", "Company is still active", "Company filed for bankruptcy",
                             "Company was acquired / merged", "Other"]
                )
                comments = st.text_area(
                    "Additional comments (optional)",
                    placeholder="Any observations about the prediction or the financial data...",
                    height=108
                )

            submitted = st.form_submit_button("Submit Feedback", use_container_width=True)

            if submitted:
                feedback_row = {
                    "timestamp":       datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "company":         company_label if company_label else "Anonymous",
                    "predicted_risk":  str(results['Risk Level'].iloc[0]) if len(results) == 1 else "Multiple",
                    "predicted_prob":  f"{raw_proba[0]*100:.2f}%" if len(results) == 1 else "Multiple",
                    "accuracy_rating": accuracy,
                    "actual_outcome":  actual_outcome,
                    "comments":        comments
                }
                feedback_path = "feedback.csv"
                feedback_df   = pd.DataFrame([feedback_row])
                if os.path.exists(feedback_path):
                    feedback_df.to_csv(feedback_path, mode='a', header=False, index=False)
                else:
                    feedback_df.to_csv(feedback_path, index=False)

                st.markdown("""
<div class="fb-submitted">
    ✅ <div><strong>Thank you for your feedback!</strong>
    Your response has been recorded and will help improve future predictions.</div>
</div>
""", unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error processing file: {e}")
