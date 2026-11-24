"""
app.py - Streamlit Interactive Lead Scoring & Prioritization Application.
Predicting Term Deposit Subscription with Ensemble Models (Pre-Contact Scenario, Dark Mode).
"""
import os
import joblib
import pandas as pd
import numpy as np
import streamlit as st

# Configure page
st.set_page_config(
    page_title="Bank Lead Prioritization Dashboard",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Dark Mode Specific CSS Adjustments
st.markdown("""
<style>
    /* Prevent title cut off and ensure clean vertical flow */
    .block-container {
        padding-top: 2.5rem !important;
        padding-bottom: 3rem !important;
        max-width: 1250px !important;
    }
    .priority-badge-high {
        background-color: rgba(34, 197, 94, 0.2);
        color: #4ADE80;
        font-weight: 700;
        padding: 6px 16px;
        border-radius: 20px;
        display: inline-block;
        font-size: 0.92rem;
        border: 1px solid #22C55E;
    }
    .priority-badge-med {
        background-color: rgba(234, 179, 8, 0.2);
        color: #FACC15;
        font-weight: 700;
        padding: 6px 16px;
        border-radius: 20px;
        display: inline-block;
        font-size: 0.92rem;
        border: 1px solid #EAB308;
    }
    .priority-badge-low {
        background-color: rgba(239, 68, 68, 0.2);
        color: #F87171;
        font-weight: 700;
        padding: 6px 16px;
        border-radius: 20px;
        display: inline-block;
        font-size: 0.92rem;
        border: 1px solid #EF4444;
    }
    .disclaimer-box {
        background-color: rgba(17, 24, 39, 0.9);
        border: 1px solid #1E3A8A;
        border-left: 5px solid #3B82F6;
        padding: 0.9rem 1.2rem;
        border-radius: 8px;
        color: #93C5FD;
        font-size: 0.88rem;
        line-height: 1.5;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    model_path = os.path.join("models", "best_model.joblib")
    if not os.path.exists(model_path):
        st.error(f"Model file not found at {model_path}. Please run src/train.py first.")
        st.stop()
    return joblib.load(model_path)


def main():
    pipeline = load_model()

    # Title & Subtitle with native Streamlit components (no cut off)
    st.title("🏦 Bank Marketing Lead Prioritization System")
    st.caption("Pre-Call Term Deposit Propensity Scoring & Lead Queue Allocation (Zero Target Leakage)")

    # Anti-Leakage Protocol Guarantee
    st.info(
        "🛡️ **Pre-Contact Protocol:** Call duration (`duration`) is strictly excluded from all model computations to eliminate target leakage. "
        "Predictions utilize only customer demographic data, core financial profile, and historical outreach logs available *before* placing calls."
    )

    # 1. Lead Input Profile Card
    with st.container(border=True):
        st.subheader("📋 Prospective Client Profile (Pre-Call Attributes)")

        col1, col2, col3 = st.columns(3, gap="medium")

        with col1:
            st.markdown("**1. Demographics**")
            age = st.slider("Age (years)", 18, 95, 38)
            job = st.selectbox("Job / Occupation", [
                "management", "technician", "blue-collar", "admin.", "services",
                "retired", "self-employed", "student", "unemployed", "entrepreneur",
                "housemaid", "Unknown"
            ])
            marital = st.selectbox("Marital Status", ["married", "single", "divorced"])
            education = st.selectbox("Education Level", ["secondary", "tertiary", "primary", "Unknown"])

        with col2:
            st.markdown("**2. Financial & Debt Profile**")
            balance = st.number_input("Average Yearly Balance (€)", -10000, 200000, 2500, 250)
            default = st.selectbox("Credit Currently in Default?", ["no", "yes"])
            housing = st.selectbox("Active Housing Loan / Mortgage?", ["yes", "no"])
            loan = st.selectbox("Active Personal Consumer Loan?", ["no", "yes"])

        with col3:
            st.markdown("**3. Campaign Outreach Parameters**")
            contact = st.selectbox("Communication Channel", ["cellular", "telephone", "Unknown"])
            month = st.selectbox("Target Contact Month", [
                "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec", "jan", "feb"
            ], index=4)
            day_of_week = st.slider("Day of the Month (1-31)", 1, 31, 15)
            campaign = st.number_input("Contacts Planned in Current Campaign", 1, 50, 1)

            hist_contact = st.radio("Contacted in Prior Marketing Campaigns?", ["No", "Yes"], horizontal=True)
            if hist_contact == "Yes":
                p_c1, p_c2 = st.columns(2)
                with p_c1:
                    pdays = st.number_input("pdays", 1, 999, 90)
                    previous = st.number_input("previous", 1, 50, 2)
                with p_c2:
                    poutcome = st.selectbox("Prior Outcome", ["success", "failure", "other", "Unknown"], index=0)
            else:
                pdays = -1
                previous = 0
                poutcome = "Unknown"

    # Lead inference
    input_df = pd.DataFrame([{
        'age': age,
        'job': job,
        'marital': marital,
        'education': education,
        'default': default,
        'balance': balance,
        'housing': housing,
        'loan': loan,
        'contact': contact,
        'day_of_week': day_of_week,
        'month': month,
        'campaign': campaign,
        'pdays': pdays,
        'previous': previous,
        'poutcome': poutcome
    }])

    prob_subscription = float(pipeline.predict_proba(input_df)[0, 1])
    decision_threshold = 0.49  # Calibrated threshold for champion Extra Trees model
    is_recommended = prob_subscription >= decision_threshold

    # 2. Predictive Results Card
    with st.container(border=True):
        st.subheader("🔍 Predictive Scoring & Strategic Queue Allocation")

        res_col1, res_col2 = st.columns([1.1, 1.0], gap="large")

        with res_col1:
            st.markdown("#### Propensity Assessment")
            m_col1, m_col2 = st.columns(2)
            with m_col1:
                st.metric(
                    label="Subscription Probability",
                    value=f"{prob_subscription * 100:.1f}%",
                    delta=f"{(prob_subscription - 0.316) * 100:+.1f}% vs Holdout Baseline (31.6%)"
                )
            with m_col2:
                st.markdown("**Priority Queue:**")
                if prob_subscription >= 0.60:
                    st.markdown('<div class="priority-badge-high">🟢 HIGH PRIORITY LEAD</div>', unsafe_allow_html=True)
                    queue_desc = "Route immediately to senior relationship specialists."
                elif prob_subscription >= decision_threshold:
                    st.markdown('<div class="priority-badge-med">🟡 MEDIUM PRIORITY LEAD</div>', unsafe_allow_html=True)
                    queue_desc = "Queue for automated dialer or standard outbound campaign."
                else:
                    st.markdown('<div class="priority-badge-low">🔴 LOW PRIORITY LEAD</div>', unsafe_allow_html=True)
                    queue_desc = "Deprioritize from voice calling; divert to digital mobile banners."
                st.caption(queue_desc)

            st.progress(min(max(prob_subscription, 0.0), 1.0))

            if is_recommended:
                st.success(
                    f"✅ **Action: Include in Outbound Queue** — Calculated propensity ({prob_subscription*100:.1f}%) "
                    f"exceeds calibrated operational cutoff ({decision_threshold*100:.1f}%)."
                )
            else:
                st.warning(
                    f"⚠️ **Action: Suppress / Defer Outbound Call** — Calculated propensity ({prob_subscription*100:.1f}%) "
                    f"is below calibrated cutoff ({decision_threshold*100:.1f}%)."
                )

        with res_col2:
            st.markdown("#### Key Drivers & Outbound Guidance")
            st.markdown("""
            - **Past Outcome (`poutcome`):** Prior campaign success provides the largest single predictive lift.
            - **Seasonality (`month`):** March, September, October, and December historically exhibit peak conversion rates.
            - **Debt Freedom (`housing` & `loan`):** Clients without housing or personal debt exhibit 2.3× higher liquid surplus.
            """)

            if poutcome == "success":
                tip_msg = "🌟 **High-Value Client:** Customer previously subscribed to a bank product. Pitch premium tiered yield."
            elif housing == "no" and loan == "no":
                tip_msg = "💡 **Zero-Debt Profile:** Prospect has no active loans. Emphasize guaranteed returns and liquidity."
            elif campaign > 3:
                tip_msg = "⚠️ **High Fatigue Alert:** Lead has been contacted multiple times this campaign. Limit call frequency."
            else:
                tip_msg = "ℹ️ **Standard Profile:** Present standard certificate of deposit options with flexible tenors."

            st.info(tip_msg)

        # Operational Disclaimer
        st.markdown("""
        <div class="disclaimer-box">
            <b>Operational Disclaimer:</b>
            This prediction represents statistical propensity estimates derived from historical Portuguese retail banking direct telemarketing data (UCI Machine Learning Repository).
            Predictions are intended solely for lead list ranking and queue prioritization, not as a factual guarantee of contract subscription.
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
