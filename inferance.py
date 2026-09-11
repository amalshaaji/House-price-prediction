from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

FEATURES = [
    "square_feet",
    "num_rooms",
    "age",
    "distance_to_city(km)",
]

MODEL_PATH = Path(__file__).parent / "house_price_model.joblib"


def predict_house_price(
    square_feet: float | int | pd.DataFrame = None,
    num_rooms: float | int = None,
    age: float | int = None,
    distance_to_city_km: float | int = None,
):
    """Predict house price from raw values or a DataFrame of input features."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Train and save the model in the notebook first."
        )

    model = joblib.load(MODEL_PATH)

    if isinstance(square_feet, pd.DataFrame):
        input_data = square_feet
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

    missing_columns = [column for column in FEATURES if column not in input_data.columns]
    if missing_columns:
        raise ValueError(f"Missing required feature columns: {missing_columns}")

    prediction = model.predict(input_data[FEATURES])[0]
    return float(prediction)
