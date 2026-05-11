"""
Customer Segmentation & Churn Engine
======================================
A decision intelligence platform that mirrors what Uber, Netflix,
Salesforce, and HubSpot run in production for customer retention.

Four pages:
  1. Segmentation Explorer  — UMAP clusters, segment profiles, bootstrap stability
  2. Churn Risk Dashboard   — Per-segment models, calibrated probabilities, SHAP
  3. Uplift Intelligence    — Persuadable identification, ROI ranking, causal ML
  4. Retention Actions      — LLM-generated intervention strategies per customer
"""

import os
import sys
import json
import warnings
import joblib

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")

# Add src to path
SRC_PATH = os.path.join(os.path.dirname(__file__), "src")
sys.path.insert(0, SRC_PATH)

PROCESSED_PATH = os.path.join(os.path.dirname(__file__), "data", "processed")
MODELS_PATH = os.path.join(os.path.dirname(__file__), "models")

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Churn Engine",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Colour palette (consistent across all charts) ──────────────────────────
SEGMENT_COLORS = {
    "Champions": "#2ECC71",
    "Loyal Customers": "#3498DB",
    "At-Risk": "#E74C3C",
    "Price Sensitive": "#F39C12",
    "Lapsed": "#95A5A6",
}

CUSTOMER_TYPE_COLORS = {
    "Persuadable": "#27AE60",
    "Sure Thing": "#2980B9",
    "Lost Cause": "#E74C3C",
    "Sleeping Dog": "#7F8C8D",
}

RISK_COLORS = {
    "High Risk": "#E74C3C",
    "Medium Risk": "#F39C12",
    "Low Risk": "#2ECC71",
}


# ─── Data Loading (cached) ──────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_parquet(os.path.join(PROCESSED_PATH, "uplift.parquet"))
    return df


@st.cache_data
def load_stability():
    path = os.path.join(MODELS_PATH, "stability.pkl")
    if os.path.exists(path):
        return joblib.load(path)
    return None


@st.cache_resource
def load_segment_models():
    path = os.path.join(MODELS_PATH, "segment_models.pkl")
    if os.path.exists(path):
        return joblib.load(path)
    return {}


def build_segment_profiles(df):
    cols = [
        "EngagementScore",
        "RecencySignal",
        "StickinessIndex",
        "SpendTrend",
        "SupportRiskScore",
        "DiscountSensitivity",
        "TenureStability",
        "Churn",
    ]
    available = [c for c in cols if c in df.columns]
    profile = df.groupby("Segment")[available].mean().round(3)
    profile["CustomerCount"] = df.groupby("Segment").size()
    profile["ChurnRate"] = df.groupby("Segment")["Churn"].mean().round(3)
    return profile


# ─── Sidebar ────────────────────────────────────────────────────────────────
def render_sidebar(df):
    st.sidebar.image("https://img.icons8.com/fluency/96/000000/target.png", width=60)
    st.sidebar.title("Churn Engine")
    st.sidebar.caption("Decision Intelligence Platform")
    st.sidebar.markdown("---")

    page = st.sidebar.radio(
        "Navigate",
        [
            "Segmentation Explorer",
            "Churn Risk Dashboard",
            "Uplift Intelligence",
            "Retention Actions",
        ],
        index=0,
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Pipeline Summary**")
    st.sidebar.metric("Total Customers", f"{len(df):,}")
    st.sidebar.metric("Segments", df["Segment"].nunique())
    st.sidebar.metric(
        "Persuadables",
        f"{(df['CustomerType'] == 'Persuadable').sum():,}",
        help="Customers who will churn AND respond to intervention",
    )
    st.sidebar.metric(
        "High Risk",
        f"{(df['RiskTier'] == 'High Risk').sum():,}",
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Architecture**")
    st.sidebar.markdown(
        "- K-Means++ segmentation\n"
        "- GMM soft assignments\n"
        "- UMAP visualization\n"
        "- Bootstrap ARI stability\n"
        "- Per-segment XGBoost\n"
        "- Isotonic calibration\n"
        "- CausalML uplift (T+S Learner)\n"
        "- Claude retention actions"
    )

    return page


# ─── Page 1: Segmentation Explorer ──────────────────────────────────────────
def page_segmentation(df):
    st.title("Customer Segmentation Explorer")
    st.markdown(
        "Customers are segmented into behavioral cohorts using **K-Means++**, "
        "validated with **Gaussian Mixture Models** (soft probability assignments), "
        "and visualized in 2D using **UMAP**. Segment stability is validated via "
        "**bootstrap Adjusted Rand Index** across 100 resamplings — the same "
        "validation approach used in production ML systems."
    )

    stability = load_stability()

    # ── KPIs ────────────────────────────────────────────────────────────────
    col1, col2, col3, col4, col5 = st.columns(5)
    for i, (seg, color) in enumerate(SEGMENT_COLORS.items()):
        count = (df["Segment"] == seg).sum()
        churn_rate = df[df["Segment"] == seg]["Churn"].mean() if count > 0 else 0
        [col1, col2, col3, col4, col5][i].metric(
            seg, f"{count:,}", f"{churn_rate:.1%} churn"
        )

    st.markdown("---")

    # ── UMAP Scatter ────────────────────────────────────────────────────────
    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.subheader("UMAP Cluster Visualization")
        color_by = st.selectbox(
            "Colour by",
            [
                "Segment",
                "Churn",
                "RiskTier",
                "CustomerType",
                "EngagementScore",
                "ChurnProbability",
                "UpliftScore",
            ],
            index=0,
        )

        if color_by == "Segment":
            fig = px.scatter(
                df,
                x="UMAP_1",
                y="UMAP_2",
                color="Segment",
                color_discrete_map=SEGMENT_COLORS,
                opacity=0.7,
                hover_data=["CustomerID", "Churn", "ChurnProbability"],
                title="Customer Behavioral Space (UMAP 2D)",
            )
        elif color_by in ["EngagementScore", "ChurnProbability", "UpliftScore"]:
            fig = px.scatter(
                df,
                x="UMAP_1",
                y="UMAP_2",
                color=color_by,
                color_continuous_scale="RdYlGn_r",
                opacity=0.7,
                hover_data=["CustomerID", "Segment"],
                title=f"Customer Behavioral Space — coloured by {color_by}",
            )
        elif color_by == "RiskTier":
            fig = px.scatter(
                df,
                x="UMAP_1",
                y="UMAP_2",
                color="RiskTier",
                color_discrete_map=RISK_COLORS,
                opacity=0.7,
                hover_data=["CustomerID", "Segment", "ChurnProbability"],
                title="Customer Behavioral Space — coloured by Risk Tier",
            )
        elif color_by == "CustomerType":
            fig = px.scatter(
                df,
                x="UMAP_1",
                y="UMAP_2",
                color="CustomerType",
                color_discrete_map=CUSTOMER_TYPE_COLORS,
                opacity=0.7,
                hover_data=["CustomerID", "Segment"],
                title="Customer Behavioral Space — coloured by Customer Type",
            )
        else:
            fig = px.scatter(
                df,
                x="UMAP_1",
                y="UMAP_2",
                color=color_by,
                opacity=0.7,
                title=f"UMAP — {color_by}",
            )

        fig.update_traces(marker=dict(size=4))
        fig.update_layout(height=500, template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.subheader("Bootstrap Stability")
        if stability:
            ari = stability["mean_ari"]
            grade = stability["grade"]
            color = (
                "#2ECC71" if ari >= 0.85 else "#F39C12" if ari >= 0.70 else "#E74C3C"
            )

            st.metric(
                "Mean ARI",
                f"{ari:.3f}",
                help="Adjusted Rand Index across 100 bootstrap resamplings",
            )
            st.metric("Std ARI", f"{stability['std_ari']:.3f}")
            st.metric("Stability Grade", grade)

            ari_scores = stability.get("ari_scores", [])
            if ari_scores:
                fig_ari = px.histogram(
                    x=ari_scores,
                    nbins=20,
                    labels={"x": "ARI Score"},
                    title="ARI Distribution (100 bootstraps)",
                    color_discrete_sequence=[color],
                )
                fig_ari.add_vline(
                    x=ari,
                    line_dash="dash",
                    line_color="black",
                    annotation_text=f"Mean={ari:.3f}",
                )
                fig_ari.update_layout(
                    height=280, template="plotly_white", showlegend=False
                )
                st.plotly_chart(fig_ari, use_container_width=True)

            st.info(
                "**What is Bootstrap ARI?**\n\n"
                "Bootstrap stability tests whether the same customer segments "
                "emerge from different random samples of the data. ARI > 0.70 "
                "means the segments are consistent — not an artifact of one "
                "random seed. This is a production-grade validation step."
            )
        else:
            st.warning("Stability data not available. Run pipeline first.")

    st.markdown("---")

    # ── Segment Profiles Heatmap ─────────────────────────────────────────────
    st.subheader("Segment Behavioral Profiles")
    profiles = build_segment_profiles(df)

    heat_cols = [
        "EngagementScore",
        "RecencySignal",
        "StickinessIndex",
        "SpendTrend",
        "SupportRiskScore",
        "DiscountSensitivity",
        "TenureStability",
        "ChurnRate",
    ]
    heat_data = profiles[[c for c in heat_cols if c in profiles.columns]]

    fig_heat = px.imshow(
        heat_data.T,
        color_continuous_scale="RdYlGn_r",
        title="Segment Profile Heatmap (darker red = higher value)",
        text_auto=".2f",
        aspect="auto",
    )
    fig_heat.update_layout(height=400, template="plotly_white")
    st.plotly_chart(fig_heat, use_container_width=True)

    # ── GMM Soft Probabilities ───────────────────────────────────────────────
    st.subheader("GMM Soft Segment Probabilities")
    st.markdown(
        "Unlike K-Means (hard assignment), GMM assigns each customer a **probability "
        "distribution across all segments**. A customer who is '60% At-Risk, 30% Lapsed' "
        "is treated differently from one who is '95% At-Risk'. This is the approach "
        "used in production for ambiguous health score boundaries."
    )
    gmm_cols = [c for c in df.columns if c.startswith("GMM_Prob_Seg")]
    if gmm_cols:
        sample_df = df.sample(min(500, len(df)), random_state=42)
        fig_gmm = px.bar(
            sample_df[["Segment"] + gmm_cols].sort_values("Segment"),
            x=sample_df.index[: len(sample_df)],
            y=gmm_cols,
            color_discrete_sequence=list(SEGMENT_COLORS.values()),
            title="GMM Soft Assignment Probabilities (sample of 500 customers)",
            labels={"value": "Probability", "variable": "Segment"},
        )
        fig_gmm.update_layout(height=300, template="plotly_white", barmode="stack")
        st.plotly_chart(fig_gmm, use_container_width=True)


# ─── Page 2: Churn Risk Dashboard ───────────────────────────────────────────
def page_churn_risk(df):
    st.title("Churn Risk Dashboard")
    st.markdown(
        "Per-segment XGBoost classifiers with **isotonic probability calibration**. "
        "Calibrated probabilities are used for ROI calculations — a raw 0.7 score "
        "doesn't mean 70% of customers churn, but a calibrated 0.7 does. "
        "This matches how Salesforce Einstein and HubSpot score customer health."
    )

    seg_models = load_segment_models()

    # ── Model Performance ───────────────────────────────────────────────────
    st.subheader("Per-Segment Model Performance")
    metrics_data = []
    for seg, model_dict in seg_models.items():
        if model_dict and "metrics" in model_dict:
            m = model_dict["metrics"]
            metrics_data.append(
                {
                    "Segment": seg,
                    "Customers": m.get("n_customers", 0),
                    "Churn Rate": f"{m.get('churn_rate', 0):.1%}",
                    "CV AUC": f"{m.get('cv_auc', 0):.3f}",
                    "CV AP": f"{m.get('cv_ap', 0):.3f}",
                    "Brier Score": f"{m.get('brier_score', 0):.4f}",
                }
            )
    if metrics_data:
        st.dataframe(
            pd.DataFrame(metrics_data).set_index("Segment"),
            use_container_width=True,
        )

    st.markdown("---")

    # ── Risk Distribution ───────────────────────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Churn Probability Distribution")
        seg_filter = st.multiselect(
            "Filter by segment",
            df["Segment"].unique().tolist(),
            default=df["Segment"].unique().tolist(),
        )
        df_filtered = df[df["Segment"].isin(seg_filter)]

        fig_hist = px.histogram(
            df_filtered,
            x="ChurnProbability",
            color="Segment",
            color_discrete_map=SEGMENT_COLORS,
            nbins=40,
            barmode="overlay",
            opacity=0.7,
            title="Calibrated Churn Probability by Segment",
            labels={"ChurnProbability": "Churn Probability"},
        )
        fig_hist.update_layout(height=380, template="plotly_white")
        st.plotly_chart(fig_hist, use_container_width=True)

    with col2:
        st.subheader("Risk Tier Breakdown")
        risk_seg = (
            df_filtered.groupby(["Segment", "RiskTier"])
            .size()
            .reset_index(name="Count")
        )
        fig_risk = px.bar(
            risk_seg,
            x="Segment",
            y="Count",
            color="RiskTier",
            color_discrete_map=RISK_COLORS,
            title="Risk Tier Distribution by Segment",
            barmode="group",
        )
        fig_risk.update_layout(height=380, template="plotly_white")
        st.plotly_chart(fig_risk, use_container_width=True)

    st.markdown("---")

    # ── SHAP Feature Importance per Segment ─────────────────────────────────
    st.subheader("Feature Importance by Segment (Gain-Based SHAP Proxy)")
    st.markdown(
        "XGBoost gain-based feature importance — normalized per segment. "
        "Different segments have different churn drivers, which is why per-segment "
        "models outperform a single global model."
    )
    selected_seg = st.selectbox("Select segment", list(seg_models.keys()))
    if selected_seg in seg_models and seg_models[selected_seg]:
        mean_abs_shap = seg_models[selected_seg]["mean_abs_shap"]
        shap_df = mean_abs_shap.reset_index()
        shap_df.columns = ["Feature", "Importance"]
        shap_df = shap_df.head(15)

        fig_shap = px.bar(
            shap_df,
            x="Importance",
            y="Feature",
            orientation="h",
            color="Importance",
            color_continuous_scale="Reds",
            title=f"Top Feature Importance — {selected_seg} Segment",
        )
        fig_shap.update_layout(
            height=400, template="plotly_white", yaxis=dict(autorange="reversed")
        )
        st.plotly_chart(fig_shap, use_container_width=True)

    st.markdown("---")

    # ── High-Risk Customer Table ─────────────────────────────────────────────
    st.subheader("High-Risk Customers")
    high_risk = df[df["RiskTier"] == "High Risk"].nlargest(50, "ChurnProbability")
    display_cols = [
        "CustomerID",
        "Segment",
        "ChurnProbability",
        "RiskTier",
        "CustomerType",
        "NetROI",
        "HourSpendOnApp",
        "DaySinceLastOrder",
        "SatisfactionScore",
        "Complain",
    ]
    display_cols = [c for c in display_cols if c in high_risk.columns]

    st.dataframe(
        high_risk[display_cols]
        .reset_index(drop=True)
        .style.background_gradient(subset=["ChurnProbability"], cmap="Reds"),
        use_container_width=True,
        height=400,
    )


# ─── Page 3: Uplift Intelligence ────────────────────────────────────────────
def page_uplift(df):
    st.title("Uplift Intelligence")
    st.markdown(
        "**Uplift modeling** (causal ML) identifies which customers will **respond** "
        "to a retention intervention — not just who will churn. Targeting the wrong "
        "customers wastes retention budget. This is the approach used by "
        "**Uber (CausalML)**, **Netflix**, and **Salesforce** in production."
    )

    # ── CLV / Cost Controls ─────────────────────────────────────────────────
    st.sidebar.markdown("---")
    st.sidebar.subheader("Business Parameters")
    avg_clv = st.sidebar.number_input(
        "Avg Customer Lifetime Value ($)", value=500, step=50
    )
    intervention_cost = st.sidebar.number_input(
        "Intervention Cost ($)", value=15, step=5
    )

    # ── The Four Quadrants ──────────────────────────────────────────────────
    st.subheader("The Four Uplift Quadrants")
    col1, col2, col3, col4 = st.columns(4)
    type_counts = df["CustomerType"].value_counts()

    col1.metric(
        "Persuadables",
        f"{type_counts.get('Persuadable', 0):,}",
        help="High churn risk + responds to intervention — TARGET THESE",
    )
    col2.metric(
        "Sure Things",
        f"{type_counts.get('Sure Thing', 0):,}",
        help="Low churn risk — would stay anyway, no action needed",
    )
    col3.metric(
        "Lost Causes",
        f"{type_counts.get('Lost Cause', 0):,}",
        help="High churn risk + won't respond — don't waste budget",
    )
    col4.metric(
        "Sleeping Dogs",
        f"{type_counts.get('Sleeping Dog', 0):,}",
        help="Low churn risk — do NOT intervene, risk triggering churn",
    )

    st.markdown("---")

    # ── Uplift Scatter ──────────────────────────────────────────────────────
    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.subheader("Churn Probability vs Uplift Score")
        fig_scatter = px.scatter(
            df,
            x="ChurnProbability",
            y="UpliftScore",
            color="CustomerType",
            color_discrete_map=CUSTOMER_TYPE_COLORS,
            opacity=0.6,
            hover_data=["CustomerID", "Segment", "NetROI"],
            title="Uplift Quadrant Map",
            labels={
                "ChurnProbability": "Churn Probability",
                "UpliftScore": "Uplift Score",
            },
        )
        fig_scatter.add_hline(
            y=0.05,
            line_dash="dash",
            line_color="gray",
            annotation_text="Uplift threshold",
        )
        fig_scatter.add_vline(
            x=0.30,
            line_dash="dash",
            line_color="gray",
            annotation_text="Churn threshold",
        )
        fig_scatter.update_traces(marker=dict(size=5))
        fig_scatter.update_layout(height=450, template="plotly_white")
        st.plotly_chart(fig_scatter, use_container_width=True)

    with col_right:
        st.subheader("ROI Analysis")
        persuadables = df[df["CustomerType"] == "Persuadable"].copy()
        persuadables["NetROI_calc"] = (
            persuadables["UpliftScore"] * avg_clv - intervention_cost
        )

        total_spend = len(persuadables) * intervention_cost
        total_retained_value = (persuadables["UpliftScore"] * avg_clv).sum()
        net_campaign_roi = total_retained_value - total_spend

        st.metric("Persuadables to Target", f"{len(persuadables):,}")
        st.metric("Total Intervention Spend", f"${total_spend:,.0f}")
        st.metric("Expected Retained Value", f"${total_retained_value:,.0f}")
        st.metric(
            "Net Campaign ROI",
            f"${net_campaign_roi:,.0f}",
            delta=f"{'+' if net_campaign_roi > 0 else ''}{net_campaign_roi / max(total_spend, 1):.1%} ROI",
        )

        # ROI by segment
        roi_by_seg = (
            persuadables.groupby("Segment")
            .agg(
                Count=("CustomerID", "count"),
                AvgUplift=("UpliftScore", "mean"),
                TotalROI=("NetROI_calc", "sum"),
            )
            .round(2)
        )
        st.dataframe(roi_by_seg, use_container_width=True)

    st.markdown("---")

    # ── Priority Intervention List ───────────────────────────────────────────
    st.subheader("Priority Intervention List (Persuadables ranked by ROI)")
    st.markdown(
        "Only **Persuadables** are shown — customers with both high churn risk "
        "and positive response to intervention. Sorted by Net ROI descending."
    )

    priority_df = df[df["CustomerType"] == "Persuadable"].copy()
    priority_df["NetROI_calc"] = (
        priority_df["UpliftScore"] * avg_clv - intervention_cost
    )
    priority_df = priority_df.nlargest(100, "NetROI_calc")

    display_cols = [
        "CustomerID",
        "Segment",
        "ChurnProbability",
        "UpliftScore",
        "NetROI_calc",
        "HourSpendOnApp",
        "DaySinceLastOrder",
        "SatisfactionScore",
        "Complain",
    ]
    display_cols = [c for c in display_cols if c in priority_df.columns]

    st.dataframe(
        priority_df[display_cols]
        .reset_index(drop=True)
        .rename(columns={"NetROI_calc": "NetROI ($)"})
        .style.background_gradient(subset=["ChurnProbability"], cmap="Reds")
        .background_gradient(subset=["UpliftScore"], cmap="Greens"),
        use_container_width=True,
        height=400,
    )

    # ── Uplift Distribution ──────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Uplift Score Distribution by Segment")
    fig_up = px.box(
        df,
        x="Segment",
        y="UpliftScore",
        color="Segment",
        color_discrete_map=SEGMENT_COLORS,
        title="Uplift Score Distribution — which segments are most responsive to intervention?",
    )
    fig_up.update_layout(height=380, template="plotly_white", showlegend=False)
    st.plotly_chart(fig_up, use_container_width=True)


# ─── Page 4: Retention Actions ──────────────────────────────────────────────
def page_retention_actions(df):
    st.title("LLM-Powered Retention Actions")
    st.markdown(
        "For each Persuadable customer, Claude analyzes their **SHAP risk factors**, "
        "**segment profile**, **churn probability**, and **uplift score** to generate "
        "a structured retention strategy: intervention type, channel, timing, message "
        "framing, and estimated ROI. This mirrors Salesforce Einstein Copilot's "
        "CSM playbook generation."
    )

    # ── API Key Input ────────────────────────────────────────────────────────
    st.sidebar.markdown("---")
    st.sidebar.subheader("Claude API Key")
    api_key = st.sidebar.text_input(
        "Anthropic API Key",
        type="password",
        placeholder="sk-ant-...",
        help="Required for LLM retention action generation",
    )
    avg_clv = st.sidebar.number_input("Customer Lifetime Value ($)", value=500, step=50)
    top_n = st.sidebar.slider("Customers to generate actions for", 1, 20, 5)

    # ── Persuadable Preview ─────────────────────────────────────────────────
    persuadables = df[df["CustomerType"] == "Persuadable"].nlargest(50, "NetROI").copy()

    st.subheader(f"Top Persuadable Customers ({len(persuadables)} shown)")
    preview_cols = [
        "CustomerID",
        "Segment",
        "ChurnProbability",
        "UpliftScore",
        "NetROI",
        "SatisfactionScore",
        "Complain",
        "HourSpendOnApp",
    ]
    preview_cols = [c for c in preview_cols if c in persuadables.columns]
    st.dataframe(
        persuadables[preview_cols].reset_index(drop=True), use_container_width=True
    )

    st.markdown("---")

    # ── Generate Actions ─────────────────────────────────────────────────────
    if st.button("Generate Retention Actions", type="primary", disabled=not api_key):
        if not api_key:
            st.error("Please enter your Anthropic API key in the sidebar.")
            return

        from retention_llm import generate_batch_retention_actions

        seg_profiles = build_segment_profiles(df)

        with st.spinner(
            f"Generating retention actions for {top_n} customers via Claude..."
        ):
            actions = generate_batch_retention_actions(
                df_uplift=persuadables,
                segment_profiles=seg_profiles,
                api_key=api_key,
                top_n=top_n,
                avg_clv=avg_clv,
            )

        st.success(f"Generated {len(actions)} retention action plans.")
        st.markdown("---")

        for action in actions:
            cid = action.get("customer_id", "N/A")
            seg = action.get("segment", "N/A")
            churn_p = action.get("churn_probability", 0)
            uplift = action.get("uplift_score", 0)
            roi = action.get("net_roi", 0)

            with st.expander(
                f"Customer {cid} | {seg} | Churn: {churn_p:.1%} | "
                f"Uplift: {uplift:+.3f} | ROI: ${roi:.0f}",
                expanded=True,
            ):
                if action.get("error"):
                    st.error(f"Error: {action['error']}")
                elif action.get("do_not_intervene_reason"):
                    st.warning(f"No intervention: {action['do_not_intervene_reason']}")
                else:
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Intervention", action.get("intervention_type", "N/A"))
                    col2.metric("Channel", action.get("channel", "N/A"))
                    col3.metric("Timing", action.get("timing", "N/A"))

                    col4, col5 = st.columns(2)
                    col4.metric("Cost", action.get("intervention_cost_estimate", "N/A"))
                    col5.metric("Confidence", action.get("confidence", "N/A"))

                    st.markdown(
                        f"**Why at risk:** {action.get('primary_risk_reason', '')}"
                    )
                    st.markdown(
                        f"**Will they respond?** {action.get('customer_receptivity', '')}"
                    )
                    st.info(
                        f"**Suggested Message:**\n\n{action.get('message_framing', '')}"
                    )
                    st.markdown(
                        f"**Expected outcome:** {action.get('expected_outcome', '')}"
                    )

    elif not api_key:
        st.info(
            "Enter your Anthropic API key in the sidebar to generate "
            "LLM-powered retention action plans for each at-risk customer."
        )

        # Show example output when no key provided
        st.markdown("### Example Output Preview")
        st.markdown("""
**Intervention:** Loyalty Reward
**Channel:** In-App Notification | **Timing:** Immediate
**Cost:** Low ($1-5) | **Confidence:** High

**Why at risk:** Customer has not placed an order in 18 days and app engagement has declined 60% from prior period.

**Will they respond?** High probability — customer has responded to in-app promotions in the past and has a 3-year tenure indicating brand affinity.

**Suggested Message:**
> "We noticed you haven't visited us lately — and we miss you! As a valued customer, you've unlocked exclusive access to this week's top deals. Plus, we've added 200 bonus loyalty points to your account."

**Expected outcome:** 35-45% probability of re-engagement within 72 hours based on historical response patterns for this segment.
        """)


# ─── Main App ────────────────────────────────────────────────────────────────
def main():
    # Check if pipeline has been run
    uplift_path = os.path.join(PROCESSED_PATH, "uplift.parquet")
    if not os.path.exists(uplift_path):
        st.error(
            "Pipeline artifacts not found. Run `python src/pipeline.py` first to build "
            "all models and cached data."
        )
        st.code("python src/pipeline.py")
        st.stop()

    df = load_data()
    page = render_sidebar(df)

    if page == "Segmentation Explorer":
        page_segmentation(df)
    elif page == "Churn Risk Dashboard":
        page_churn_risk(df)
    elif page == "Uplift Intelligence":
        page_uplift(df)
    elif page == "Retention Actions":
        page_retention_actions(df)


if __name__ == "__main__":
    main()
