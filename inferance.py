from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union

import joblib
import pandas as pd

FEATURES = [
    "square_feet",
    "num_rooms",
    "age",
    "distance_to_city(km)",
]

MODEL_PATH = Path(__file__).parent / "house_price_model.joblib"


def load_model() -> Any:
    """Load the pre-trained linear regression model artifact."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found at {MODEL_PATH}. "
            "Please run the training notebook 'House_prediction.ipynb' first to train and export the model."
        )
    return joblib.load(MODEL_PATH)


def get_model_details() -> Dict[str, Any]:
    """Extract model coefficients and intercept for inspection."""
    model = load_model()
    intercept = getattr(model, "intercept_", 0.0)
    coefficients = getattr(model, "coef_", [])

    feature_coefs = dict(zip(FEATURES, coefficients))
    return {
        "features": FEATURES,
        "intercept": float(intercept),
        "coefficients": {k: float(v) for k, v in feature_coefs.items()},
    }


def predict_house_price(
    square_feet: Optional[Union[float, int, pd.DataFrame]] = None,
    num_rooms: Optional[Union[float, int]] = None,
    age: Optional[Union[float, int]] = None,
    distance_to_city_km: Optional[Union[float, int]] = None,
) -> float:
    """Predict house price from feature inputs or a pandas DataFrame."""
    model = load_model()

    if isinstance(square_feet, pd.DataFrame):
        input_data = square_feet.copy()
    else:
        if None in (square_feet, num_rooms, age, distance_to_city_km):
            raise ValueError(
                "Provide either a DataFrame or all four numeric inputs: "
                "square_feet, num_rooms, age, distance_to_city_km"
            )
        input_data = pd.DataFrame(
            [
                {
                    "square_feet": float(square_feet),
                    "num_rooms": float(num_rooms),
                    "age": float(age),
                    "distance_to_city(km)": float(distance_to_city_km),
                }
            ]
        )

    missing_columns = [col for col in FEATURES if col not in input_data.columns]
    if missing_columns:
        raise ValueError(f"Missing required feature columns: {missing_columns}")

    prediction = model.predict(input_data[FEATURES])[0]
    return float(prediction)
