# 🌧️ Seattle Weather Prediction Using Ensemble Methods

## 📌 Objective
The objective of this project is to build and evaluate ensemble machine learning models to predict whether it will **rain or not** in Seattle using the **Seattle Weather dataset**.

---

## 🧪 Dataset Overview

The dataset includes weather information for Seattle, including temperature, precipitation, and weather conditions. The target variable is `weather`, which will be converted into a binary classification problem:  
- `rain` = rain  
- `no rain` = all other weather conditions

---

## 🛠️ Data Preprocessing & Feature Engineering

Key preprocessing steps:
- **Binary Target Transformation**: Converted `weather` into `"rain"` or `"no rain"` labels
- **Feature Engineering**: Derived `month` from the `date` column using Python’s `datetime` module:
  - `month`
  - (Other relevant time-based features may be added)

- **Train/Test Split**: Used an 80/20 split with `stratify=y` to maintain class balance

---

## 🤖 Models Implemented

### 1. **BaggingClassifier**
- Tuned parameters:
  - `n_estimators`
  - `max_samples`
  - `max_features`
- Evaluated using `oob_score_`
- Multiple combinations tried to maximize the **OOB Score**

### 2. **RandomForestClassifier**
- Tuned parameters:
  - `n_estimators`
  - `max_features`
- Evaluated using `oob_score_`
- Focused on finding the highest-performing model

### 3. **AdaBoostClassifier**
- Tuned parameters:
  - `n_estimators`
  - `learning_rate`
- Evaluated using `.score()` on training data
- Multiple models tested to identify the best performing configuration

### 4. **XGBClassifier**
- Tuned parameters:
  - `n_estimators`
  - `max_depth`
  - `learning_rate (eta)`
- Evaluated using `.score()` on training data
- XGBoost used for its advanced boosting capabilities

---

## 📊 Model Evaluation

The **best-performing model from each method** (Bagging, RF, AdaBoost, XGBoost) was selected based on their respective metrics (`oob_score_` or `.score()`).

These models were then used to make predictions on the **test set**, and the following evaluation metrics were generated for each:
- **Confusion Matrix**
- **Accuracy**
- **F1 Score**

---

## 🧠 Analysis & Comparison

Each model’s performance on the test set was analyzed to determine:

- Which model performed best overall
- Differences in precision, recall, and F1 between methods
- Strengths/weaknesses of bagging vs. boosting approaches in this context
- Whether the hypothesized best model actually performed the best

---

## ⚙️ Tools & Libraries

- **Python**
- **Pandas**, **NumPy** – Data manipulation
- **Matplotlib**, **Seaborn** – Visualization
- **Scikit-learn** – Bagging, Random Forest, AdaBoost
- **XGBoost** – Extreme Gradient Boosting implementation
- **Datetime** – Feature engineering from date
