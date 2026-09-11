import pandas as pd
import streamlit as st

from inferance import get_model_details, predict_house_price

# =========================================================
# PAGE CONFIGURATION & THEME INTEGRATION
# =========================================================

st.set_page_config(
    page_title="ValuAI · House Valuation Engine",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        .block-container {
            max-width: 1180px;
            padding-top: 2.5rem;
            padding-bottom: 3.5rem;
        }

        .brand-badge {
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            background: linear-gradient(135deg, #059669 0%, #10b981 100%);
            color: #ffffff;
            box-shadow: 0 4px 14px rgba(5, 150, 105, 0.25);
            padding: 0.45rem 1.1rem;
            border-radius: 30px;
            font-size: 0.88rem;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-top: 1.2rem;
            margin-bottom: 1.2rem;
        }

        .valuation-banner {
            background: linear-gradient(135deg, #059669 0%, #047857 100%);
            color: #ffffff;
            border: 1px solid rgba(5, 150, 105, 0.2);
            border-radius: 20px;
            padding: 2.2rem 2rem;
            text-align: center;
            box-shadow: 0 12px 30px rgba(5, 150, 105, 0.18);
            margin: 1.5rem 0 2rem 0;
        }

        .valuation-heading {
            font-size: 0.88rem;
            font-weight: 700;
            color: rgba(255, 255, 255, 0.85);
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-bottom: 0.4rem;
        }

        .valuation-value {
            font-size: clamp(2.8rem, 6vw, 4.4rem);
            font-weight: 850;
            color: #ffffff;
            line-height: 1.05;
            letter-spacing: -0.04em;
            margin-bottom: 0.6rem;
        }

        .valuation-meta {
            font-size: 0.95rem;
            color: rgba(255, 255, 255, 0.95);
            font-weight: 500;
        }

        .footer-credits {
            text-align: center;
            color: #64748b;
            font-size: 0.85rem;
            margin-top: 4rem;
            padding-top: 1.5rem;
            border-top: 1px solid #e2e8f0;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


def format_inr(value: float) -> str:
    """Format price in Indian Rupees standard format."""
    return f"₹{value:,.0f}"


# =========================================================
# HEADER SECTION
# =========================================================

st.markdown(
    """
    <div class="brand-badge">
        <span>⚡ Real-Estate Machine Learning v1.0</span>
    </div>
    """,
    unsafe_allow_html=True,
)

st.title(":material/real_estate_agent: House Price Valuation Engine")
st.caption(
    "Predict property values instantly with pre-trained Linear Regression and transparent feature weighting."
)

st.divider()


# =========================================================
# MODEL METADATA CHECK
# =========================================================

try:
    model_info = get_model_details()
    model_loaded = True
except FileNotFoundError:
    model_loaded = False
    model_info = None

if not model_loaded:
    st.error(":material/warning: Pre-trained model artifact missing!")
    st.info(
        "Please execute **`House_prediction.ipynb`** to train the machine learning model "
        "and generate **`house_price_model.joblib`**."
    )
    st.stop()


# =========================================================
# SIDEBAR CONTROLS & PRESETS
# =========================================================

with st.sidebar:
    st.markdown("### :material/tune: Property Presets")
    st.caption("Quickly populate sample property metrics:")

    preset = st.segmented_control(
        "Property archetype",
        options=["Urban Flat", "Suburban Home", "Luxury Estate", "Custom"],
        default="Suburban Home",
        help="Select a preset to load typical property metrics.",
    )

    if preset == "Urban Flat":
        def_sqft, def_rooms, def_age, def_dist = 850, 2, 4, 3.5
    elif preset == "Luxury Estate":
        def_sqft, def_rooms, def_age, def_dist = 4200, 6, 2, 5.0
    elif preset == "Custom":
        def_sqft, def_rooms, def_age, def_dist = 1800, 3, 5, 10.0
    else:  # Suburban Home
        def_sqft, def_rooms, def_age, def_dist = 2200, 4, 10, 12.0

    st.divider()

    st.markdown("### :material/info: Model Information")
    st.markdown("- **Algorithm**: Multiple Linear Regression")
    st.markdown("- **Features**: Area, Rooms, Age, Distance")
    st.markdown("- **Target**: Market Price (₹)")

    st.divider()
    st.caption("📦 Model artifact: `house_price_model.joblib`")


# =========================================================
# PROPERTY SPECIFICATIONS INPUT FORM
# =========================================================

st.subheader(":material/edit_note: Property Parameters")

with st.form("valuation_input_form"):
    c1, c2 = st.columns(2, gap="medium")

    with c1:
        with st.container(border=True):
            st.markdown("##### :material/square_foot: Floor Plan & Layout")

            square_feet = st.number_input(
                "Total Built-up Area (sq. ft.)",
                min_value=300,
                max_value=12000,
                value=int(def_sqft),
                step=100,
                help="Total internal floor space in square feet.",
            )

            num_rooms = st.number_input(
                "Number of Rooms",
                min_value=1,
                max_value=15,
                value=int(def_rooms),
                step=1,
                help="Includes bedrooms and primary living rooms.",
            )

    with c2:
        with st.container(border=True):
            st.markdown("##### :material/location_on: Age & Proximity")

            age = st.number_input(
                "Property Age (years)",
                min_value=0,
                max_value=100,
                value=int(def_age),
                step=1,
                help="Years since property construction was completed.",
            )

            distance_to_city = st.number_input(
                "Distance to City Center (km)",
                min_value=0.0,
                max_value=100.0,
                value=float(def_dist),
                step=0.5,
                help="Road distance to the city business hub in kilometers.",
            )

    submitted = st.form_submit_button(
        ":material/auto_awesome: Calculate Market Valuation",
        type="primary",
    )


# Perform calculation
try:
    predicted_val = predict_house_price(
        square_feet=square_feet,
        num_rooms=num_rooms,
        age=age,
        distance_to_city_km=distance_to_city,
    )
except Exception as err:
    st.error(f"Valuation error: {err}")
    st.stop()


# =========================================================
# VALUATION HERO BANNER & KPI METRICS
# =========================================================

if predicted_val < 0:
    st.warning(
        "Notice: The combination of inputs resulted in a negative estimate. "
        "Please adjust inputs towards typical property specifications."
    )
else:
    low_band = predicted_val * 0.965
    high_band = predicted_val * 1.035
    price_per_sqft = predicted_val / square_feet if square_feet > 0 else 0

    st.markdown(
        f"""
        <div class="valuation-banner">
            <div class="valuation-heading">Estimated Market Valuation</div>
            <div class="valuation-value">{format_inr(predicted_val)}</div>
            <div class="valuation-meta">
                Estimated Band: <b>{format_inr(low_band)}</b> – <b>{format_inr(high_band)}</b>
                &nbsp;•&nbsp; Unit Rate: <b>₹{price_per_sqft:,.0f} / sq. ft.</b>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.container(border=True):
        st.markdown("##### :material/analytics: Property Metric Summary")
        m1, m2, m3, m4 = st.columns(4)

        with m1:
            st.metric("Total Area", f"{square_feet:,} sq. ft.")
        with m2:
            st.metric("Rooms", f"{num_rooms}")
        with m3:
            st.metric("Age", f"{age} yrs")
        with m4:
            st.metric("City Distance", f"{distance_to_city:g} km")


st.divider()


# =========================================================
# MODEL INSPECTION & ANALYSIS TABS
# =========================================================

st.subheader(":material/insights: Model Parameters & Transparency")

tab1, tab2, tab3 = st.tabs(
    [
        ":material/bar_chart: Feature Weights",
        ":material/functions: Regression Equation",
        ":material/schema: Pipeline Architecture",
    ]
)

with tab1:
    st.markdown("##### Feature Impact Breakdown")
    st.caption("Weight coefficients extracted from pre-trained Linear Regression artifact:")

    intercept = model_info["intercept"]
    coefs = model_info["coefficients"]

    coef_rows = []
    for feat, weight in coefs.items():
        impact = "Positive (+)" if weight >= 0 else "Negative (-)"
        coef_rows.append(
            {
                "Feature": feat,
                "Weight (Coefficient)": f"{weight:+,.2f}",
                "Impact Direction": impact,
            }
        )

    coef_df = pd.DataFrame(coef_rows)
    st.dataframe(coef_df, hide_index=True)

with tab2:
    st.markdown("##### Mathematical Valuation Formula")
    eq_parts = [f"{intercept:,.2f}"]
    for feat, w in coefs.items():
        sign = "+" if w >= 0 else "-"
        eq_parts.append(f"{sign} ({abs(w):,.2f} × {feat})")

    equation_str = "Price = " + " ".join(eq_parts)
    st.code(equation_str, language="text")
    st.caption(f"Baseline Model Intercept Constant: **₹{intercept:,.2f}**")

with tab3:
    st.markdown("##### Technical Architecture")
    st.markdown(
        """
        - **Jupyter Notebook**: Model training, EDA, evaluation, and serialization in `House_prediction.ipynb`.
        - **Model Artifact**: Serialized `house_price_model.joblib` binary.
        - **Inference Engine**: Lightweight prediction module in `inferance.py`.
        - **SaaS Interface**: High-performance Streamlit UI configured in `app.py` and `.streamlit/config.toml`.
        """
    )


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    """
    <div class="footer-credits">
        ValuAI · Powered by Streamlit, Scikit-Learn & Python
    </div>
    """,
    unsafe_allow_html=True,
)
