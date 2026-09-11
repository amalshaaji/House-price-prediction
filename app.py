from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from inferance import predict_house_price


# =========================================================
# CONFIG
# =========================================================

st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

FEATURES = [
    "square_feet",
    "num_rooms",
    "age",
    "distance_to_city(km)",
]
TARGET = "price"

# Keep the dataset inside the repository for deployment.
DEFAULT_DATASET = Path(__file__).parent / "house_prices_dataset.csv"


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    '''
    <style>
        .block-container {
            max-width: 1250px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }

        .hero h1 {
            font-size: 3rem;
            font-weight: 800;
            letter-spacing: -0.04em;
            margin-bottom: 0.25rem;
        }

        .hero p {
            font-size: 1.08rem;
            opacity: 0.7;
        }

        .prediction-card {
            margin: 1.5rem 0;
            padding: 2.25rem 1.5rem;
            border-radius: 20px;
            text-align: center;
            border: 1px solid rgba(16, 185, 129, 0.35);
            background: linear-gradient(
                135deg,
                rgba(16, 185, 129, 0.12),
                rgba(59, 130, 246, 0.08)
            );
            box-shadow: 0 12px 35px rgba(0, 0, 0, 0.08);
        }

        .prediction-label {
            font-size: 0.95rem;
            font-weight: 600;
            opacity: 0.7;
            margin-bottom: 0.5rem;
        }

        .prediction-value {
            font-size: clamp(2.5rem, 6vw, 4.4rem);
            font-weight: 850;
            line-height: 1.05;
            letter-spacing: -0.05em;
        }

        [data-testid="stMetric"] {
            border: 1px solid rgba(128, 128, 128, 0.18);
            border-radius: 14px;
            padding: 1rem;
        }

        .footer {
            text-align: center;
            opacity: 0.6;
            font-size: 0.85rem;
            padding-top: 2rem;
        }
    </style>
    ''',
    unsafe_allow_html=True,
)


# =========================================================
# DATA + MODEL
# =========================================================

def format_currency(value: float) -> str:
    return f"₹{value:,.0f}"


@st.cache_data
def load_dataset(path_string: str):
    path = Path(path_string)
    if not path.exists():
        return None
    return pd.read_csv(path)


@st.cache_data
def load_uploaded_dataset(uploaded_file):
    return pd.read_csv(uploaded_file)


@st.cache_resource
def train_model(df: pd.DataFrame):
    required = FEATURES + [TARGET]

    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(
            "Dataset is missing required columns: " + ", ".join(missing)
        )

    data = df[required].copy()

    for column in required:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    data = data.dropna()

    if len(data) < 10:
        raise ValueError("Not enough valid rows to train the model.")

    X = data[FEATURES]
    y = data[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)

    metrics = {
        "r2": r2_score(y_test, y_pred),
        "mae": mean_absolute_error(y_test, y_pred),
        "mse": mse,
        "rmse": mse ** 0.5,
        "train_rows": len(X_train),
        "test_rows": len(X_test),
    }

    return model, metrics


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '''
    <div class="hero">
        <h1>🏠 House Price Predictor</h1>
        <p>Estimate house prices using Multiple Linear Regression.</p>
    </div>
    ''',
    unsafe_allow_html=True,
)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:
    st.header("⚙️ Configuration")

    uploaded_file = st.file_uploader(
        "Upload dataset",
        type=["csv"],
        help="CSV must contain the four features and the price column.",
    )

    if uploaded_file is not None:
        dataset = load_uploaded_dataset(uploaded_file)
        source_name = uploaded_file.name
    else:
        dataset = load_dataset(str(DEFAULT_DATASET))
        source_name = DEFAULT_DATASET.name

    if dataset is None:
        st.warning("Dataset not found. Upload a CSV file to continue.")
        st.stop()

    st.success(f"Loaded: {source_name}")

    st.divider()

    st.subheader("Dataset")
    st.write(f"Rows: **{len(dataset):,}**")
    st.write(f"Columns: **{len(dataset.columns):,}**")

    st.divider()

    st.subheader("Required features")
    for feature in FEATURES:
        st.caption(f"• {feature}")
    st.caption(f"• {TARGET}")


# =========================================================
# VALIDATION + TRAINING
# =========================================================

required_columns = FEATURES + [TARGET]
missing_columns = [
    column for column in required_columns if column not in dataset.columns
]

if missing_columns:
    st.error("Missing columns: " + ", ".join(missing_columns))
    st.stop()

try:
    model, metrics = train_model(dataset)
except ValueError as error:
    st.error(str(error))
    st.stop()


# =========================================================
# PERFORMANCE
# =========================================================

st.subheader("📊 Model Performance")
st.caption("Metrics calculated on the unseen 20% test set.")

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric("R² Score", f"{metrics['r2']:.3f}")

with c2:
    st.metric("MAE", format_currency(metrics["mae"]))

with c3:
    st.metric("RMSE", format_currency(metrics["rmse"]))

with c4:
    st.metric("Test Samples", f"{metrics['test_rows']:,}")


st.divider()


# =========================================================
# PREDICTION
# =========================================================

st.subheader("🏡 Predict a New House")
st.caption("Enter the property details and generate an estimated price.")

with st.form("prediction_form"):
    c1, c2 = st.columns(2)

    with c1:
        square_feet = st.number_input(
            "Square Feet",
            min_value=100,
            max_value=10000,
            value=2000,
            step=100,
            help="Total built-up area.",
        )

        num_rooms = st.number_input(
            "Number of Rooms",
            min_value=1,
            max_value=20,
            value=4,
            step=1,
        )

    with c2:
        age = st.number_input(
            "House Age (Years)",
            min_value=0,
            max_value=100,
            value=5,
            step=1,
        )

        distance_to_city = st.number_input(
            "Distance to City (km)",
            min_value=0.0,
            max_value=100.0,
            value=10.0,
            step=0.5,
        )

    submitted = st.form_submit_button(
        "🔮 Predict House Price",
        use_container_width=True,
        type="primary",
    )


if submitted:
    user_input = pd.DataFrame(
        {
            "square_feet": [square_feet],
            "num_rooms": [num_rooms],
            "age": [age],
            "distance_to_city(km)": [distance_to_city],
        }
    )

    predicted_price = predict_house_price(user_input)

    if predicted_price < 0:
        st.warning(
            "The model returned a negative value. "
            "Please review the dataset and input ranges."
        )
    else:
        st.markdown(
            f'''
            <div class="prediction-card">
                <div class="prediction-label">Estimated House Price</div>
                <div class="prediction-value">{format_currency(predicted_price)}</div>
            </div>
            ''',
            unsafe_allow_html=True,
        )

        st.subheader("🏠 Property Summary")

        s1, s2, s3, s4 = st.columns(4)

        with s1:
            st.metric("Area", f"{square_feet:,} sq ft")
        with s2:
            st.metric("Rooms", str(num_rooms))
        with s3:
            st.metric("Age", f"{age} years")
        with s4:
            st.metric("Distance", f"{distance_to_city:g} km")


# =========================================================
# MODEL INSIGHTS
# =========================================================

st.divider()
st.subheader("🤖 Model Insights")

tab1, tab2, tab3 = st.tabs(
    ["Feature Coefficients", "Regression Equation", "Dataset Preview"]
)

with tab1:
    coefficient_df = pd.DataFrame(
        {
            "Feature": FEATURES,
            "Coefficient": model.coef_,
        }
    )
    coefficient_df["Coefficient"] = coefficient_df["Coefficient"].round(4)

    st.dataframe(
        coefficient_df,
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "A coefficient estimates the change in predicted price for a "
        "one-unit change in that feature, while other features remain constant."
    )


with tab2:
    equation = f"Price = {model.intercept_:.4f}"

    for feature, coefficient in zip(FEATURES, model.coef_):
        sign = "+" if coefficient >= 0 else "-"
        equation += f" {sign} {abs(coefficient):.4f} × {feature}"

    st.code(equation, language="text")
    st.write(f"**Intercept:** `{model.intercept_:.4f}`")


with tab3:
    st.dataframe(
        dataset.head(10),
        use_container_width=True,
        hide_index=True,
    )


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    '''
    <div class="footer">
        Built with Python · Pandas · Scikit-learn · Streamlit
        <br>
        House Price Prediction · Machine Learning Project
    </div>
    ''',
    unsafe_allow_html=True,
)
