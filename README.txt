# House Price Prediction — Multiple Linear Regression

A Machine Learning regression project that predicts house prices from multiple property features using **Multiple Linear Regression** with Python and Scikit-learn.

The project demonstrates an end-to-end supervised learning workflow, including dataset preparation, feature/target selection, train-test splitting, model training, prediction, evaluation, coefficient analysis, and inference on new data.

---

## Project Overview

**Problem:** Predict the price of a house based on its characteristics.

**Approach:** Train a Multiple Linear Regression model using historical house data and evaluate its performance on previously unseen test data.

### Features

| Feature                | Description            |
| ---------------------- | ---------------------- |
| `square_feet`          | House size             |
| `num_rooms`            | Number of rooms        |
| `age`                  | Age of the house       |
| `distance_to_city(km)` | Distance from the city |

### Target

| Target  | Description |
| ------- | ----------- |
| `price` | House price |

---

## Machine Learning Approach

This project uses **Multiple Linear Regression**, a supervised learning algorithm for predicting continuous numerical values.

The model learns the relationship:

```text
y = b + m₁x₁ + m₂x₂ + m₃x₃ + m₄x₄
```

Where:

* `y` = predicted house price
* `b` = model intercept
* `m₁ ... m₄` = learned coefficients
* `x₁ ... x₄` = input features

For this project:

```text
Price =
Intercept
+ m₁ × square_feet
+ m₂ × num_rooms
+ m₃ × age
+ m₄ × distance_to_city
```

---

## ML Pipeline

```text
Raw Dataset
     │
     ▼
Feature Selection
     │
     ▼
Target Selection
     │
     ▼
Train / Test Split
     │
     ├───────────────┐
     ▼               ▼
Training Data     Test Data
     │               │
     ▼               │
Linear Regression   │
     │               │
     ▼               │
Trained Model       │
     │               │
     └───────┬───────┘
             ▼
        Predictions
             │
             ▼
    Model Evaluation
             │
             ▼
     New House Inference
```

---

## Dataset Preparation

The feature matrix `X` contains the input variables:

```python
X = Data_set[
    [
        "square_feet",
        "num_rooms",
        "age",
        "distance_to_city(km)"
    ]
]
```

The target variable `y` contains the value the model is expected to predict:

```python
y = Data_set["price"]
```

---

## Train-Test Split

The dataset is divided into training and testing subsets:

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

### Split Configuration

* **80%** → Training data
* **20%** → Testing data
* `random_state=42` → Reproducible split

The model learns only from the training data and is evaluated using the unseen testing data.

---

## Model Training

```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(X_train, y_train)
```

`fit()` estimates the model's intercept and coefficients from the training dataset.

---

## Prediction

Predictions are generated using the test features:

```python
y_pred = model.predict(X_test)
```

The resulting `y_pred` contains the model's predicted prices for the test houses.

---

## Actual vs Predicted

To inspect individual predictions:

```python
comparison = pd.DataFrame({
    "Actual Price": y_test.values,
    "Predicted Price": y_pred
})

print(comparison.head(10))
```

Example:

| Actual Price | Predicted Price |
| -----------: | --------------: |
|         50.0 |            52.3 |
|         70.0 |            68.7 |
|         45.0 |            47.1 |
|         90.0 |            87.9 |

This provides a direct comparison between the ground-truth values and model predictions.

---

## Model Evaluation

The model is evaluated using standard regression metrics.

### Mean Absolute Error — MAE

```python
mae = mean_absolute_error(y_test, y_pred)
```

Measures the average absolute difference between actual and predicted values.

### Mean Squared Error — MSE

```python
mse = mean_squared_error(y_test, y_pred)
```

Penalizes larger prediction errors more heavily because the errors are squared.

### Root Mean Squared Error — RMSE

```python
rmse = mse ** 0.5
```

RMSE is the square root of MSE and expresses the error in the same unit as the target.

### R² Score

```python
r2 = r2_score(y_test, y_pred)
```

Measures the proportion of target variance explained by the regression model.

---

## Model Interpretation

The learned intercept and feature coefficients can be inspected:

```python
print("Intercept:", model.intercept_)

for feature, coefficient in zip(X.columns, model.coef_):
    print(feature, ":", coefficient)
```

Example:

```text
Intercept: 10.5

square_feet : 0.05
num_rooms : 8.2
age : -0.7
distance_to_city(km) : -1.5
```

### Interpretation

* **Positive coefficient** → Increasing the feature increases the predicted price, assuming other features remain constant.
* **Negative coefficient** → Increasing the feature decreases the predicted price, assuming other features remain constant.
* **Intercept** → Baseline value represented by the regression equation.

---

## Predicting a New House

After training, the model can be used for inference on new data.

```python
new_house = pd.DataFrame({
    "square_feet": [2400],
    "num_rooms": [4],
    "age": [5],
    "distance_to_city(km)": [8]
})

new_prediction = model.predict(new_house)

print("Predicted House Price:", new_prediction[0])
```

The model uses the four input features to generate a predicted house price.

---

## Project Structure

```text
house-price-prediction/
│
├── house_price.csv
├── House_price_predict.ipynb
├── README.md
└── requirements.txt
```

---

## Installation

Clone the repository and install the required dependencies:

```bash
git clone <repository-url>
cd house-price-prediction
```

Install dependencies:

```bash
pip install pandas scikit-learn jupyter
```

Or install from `requirements.txt`:

```bash
pip install -r requirements.txt
```

---

## Requirements

```text
pandas
scikit-learn
jupyter
```

---

## Running the Project

Launch Jupyter Notebook:

```bash
jupyter notebook
```

Open:

```text
House_price_predict.ipynb
```

Run the notebook cells sequentially to:

1. Load the dataset
2. Inspect the data
3. Select features and target
4. Split the dataset
5. Train the regression model
6. Generate predictions
7. Evaluate performance
8. Inspect model coefficients
9. Predict a new house price

---

## Key Concepts Demonstrated

**Machine Learning**

* Supervised Learning
* Regression
* Multiple Linear Regression
* Model Training
* Model Prediction
* Train-Test Split
* Model Evaluation
* Feature/Target Selection

**Python**

* Pandas
* DataFrames
* Data preprocessing
* Data manipulation

**Scikit-learn**

* `train_test_split`
* `LinearRegression`
* `mean_absolute_error`
* `mean_squared_error`
* `r2_score`

**Model Interpretation**

* Intercept
* Regression coefficients
* Actual vs predicted analysis

---

## Limitations

This project uses a Linear Regression model, which assumes a linear relationship between the selected features and house prices.

Real-world house prices can also depend on factors such as:

* Location
* Property type
* Number of bathrooms
* Land area
* Neighborhood
* Amenities
* Market conditions
* Economic factors

Therefore, the model should be treated as a learning project and baseline regression model rather than a production-grade property valuation system.

---

## Future Improvements

Potential improvements include:

* Data cleaning and validation
* Missing-value handling
* Outlier detection
* Feature scaling
* Feature engineering
* Exploratory Data Analysis (EDA)
* Cross-validation
* Hyperparameter/model comparison
* Residual analysis
* Regularized regression
* Random Forest Regression
* Gradient Boosting / XGBoost
* Feature importance analysis
* Model deployment through an API or web application

---

## Learning Outcomes

This project strengthened practical understanding of the complete basic regression workflow:

```text
Data
  ↓
Features + Target
  ↓
Train/Test Split
  ↓
Model Training
  ↓
Prediction
  ↓
Evaluation
  ↓
Interpretation
  ↓
Inference
```

It serves as a foundation for progressing from basic Machine Learning toward more advanced **Machine Learning and AI Engineering** projects.

---

## Author

**Amal Shaji**

BCA Student | AI/ML & Software Engineering

Focused on building practical skills in:

```text
Python
Machine Learning
Artificial Intelligence
Data Structures & Algorithms
Software Development
```

---

## License

This project is intended for educational and portfolio purposes.
