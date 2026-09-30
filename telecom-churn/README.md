# 📊 Telecom Customer Churn Prediction

> **Business Objective:** Predict customers likely to churn, identify key factors influencing churn, and help the telecom company take proactive retention actions to reduce customer loss and improve revenue.

---

## 📁 Project Structure

```
telecom-churn/
├── churn_model.ipynb               # Full EDA, modelling, and analysis notebook
├── churn_logistic_regression.joblib  # Saved model artifact
├── main.py                         # FastAPI backend
├── streamlit_app.py                # Streamlit frontend UI
├── model_prediction.py             # ChurnPredictor class
├── data_ingestion.py               # DataIngestion class
├── pyproject.toml                  # Project dependencies (uv)
└── README.md
```

---

## 🚀 Running Locally

You need **two terminals** inside the `telecom-churn/` folder.

**Terminal 1 — FastAPI backend**
```bash
uv sync
uv run uvicorn main:app --reload
```

**Terminal 2 — Streamlit UI**
```bash
uv run streamlit run streamlit_app.py
```

| Service | URL |
|---|---|
| 🖥️ Streamlit UI | http://localhost:8501 |
| ⚡ FastAPI Docs | http://localhost:8000/docs |
| ❤️ Health Check | http://localhost:8000/health |

---

## 📦 Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.12+ |
| ML | scikit-learn, XGBoost |
| API | FastAPI + Uvicorn |
| UI | Streamlit |
| Serialization | joblib |
| Package Manager | uv |

---

## 📂 Dataset

- **Source:** IBM Telco Customer Churn dataset (`WA_Fn-UseC_-Telco-Customer-Churn.csv`)
- **Size:** 7,043 customers × 21 features
- **Target:** `Churn` — binary (Yes / No)
- **Class Imbalance:** ~26% churn, ~74% no-churn
- **Missing Values:** `TotalCharges` had blank entries → filled with column mean

### Features Used

| Type | Features |
|---|---|
| Numerical | `tenure`, `MonthlyCharges`, `TotalCharges` |
| Nominal (OHE) | `partner`, `dependents`, `multipleLines`, `internetService`, `onlineSecurity`, `onlineBackup`, `deviceProtection`, `techSupport`, `streamingTV`, `streamingMovies`, `paperlessBilling`, `paymentMethod` |
| Ordinal | `contract` (Month-to-month → One year → Two year) |
| Dropped | `customerID` (identifier), `gender`, `phoneService` (no significant chi-square with churn) |

---

## 🔍 Exploratory Data Analysis

### Numerical Feature Insights

| Feature | Key Finding |
|---|---|
| **Tenure** | Bimodal — customers are either brand new or very loyal. No outliers. |
| **Monthly Charges** | Most customers pay **\$35–\$90/month**; median ~\$70. Wide spread. |
| **Total Charges** | Right-skewed — majority have low total charges; small loyal segment has very high totals. |

**Correlations:**

| Pair | Correlation | Interpretation |
|---|---|---|
| Tenure ↔ Total Charges | **0.82** (strong) | Longer-staying customers naturally accumulate more spend |
| Monthly Charges ↔ Total Charges | **0.65** (moderate) | Higher monthly bills lead to higher lifetime value |
| Tenure ↔ Monthly Charges | **0.25** (weak) | Bill amount doesn't predict how long someone stays |

### Churn vs Numerical Features

- **Tenure:** Churned customers have significantly **lower tenure** — newer customers churn more.
- **Monthly Charges:** Churned customers have **higher monthly charges** on average.
- **Total Charges:** Staying customers have **higher total charges**, driven by longer tenure.

### Chi-Square Test Results (Categorical Features vs Churn)

| Feature | Significant? | Chi² Statistic |
|---|---|---|
| Gender | ❌ No | 0.484 |
| PhoneService | ❌ No | 0.915 |
| SeniorCitizen | ✅ Yes | 159.43 |
| Partner | ✅ Yes | 158.73 |
| Dependents | ✅ Yes | 189.13 |
| MultipleLines | ✅ Yes | 11.33 |
| InternetService | ✅ Yes | 732.31 |
| OnlineSecurity | ✅ Yes | 849.99 |
| OnlineBackup | ✅ Yes | 601.81 |
| Contract | ✅ Yes | — |
| PaymentMethod | ✅ Yes | — |
| TechSupport | ✅ Yes | — |

> `Gender` and `PhoneService` were excluded from modelling — no significant association with churn at 95% confidence.

### Key Categorical Observations

- **Contract:** Month-to-month is the most common type and strongest churn driver.
- **Internet Service:** Fiber optic customers churn the most.
- **Online Security:** Customers without it churn at higher rates.
- **Payment Method:** Electronic check users show the highest churn rate.

---

## 🛠️ Data Preprocessing

| Step | Detail |
|---|---|
| Feature/Target split | All columns except `churn` |
| Target encoding | `No → 0`, `Yes → 1` |
| Nominal encoding | One-Hot Encoding (`handle_unknown="ignore"`) |
| Ordinal encoding | `contract` → OrdinalEncoder with explicit category order |
| Numerical scaling | StandardScaler on `tenure`, `monthlycharges`, `totalcharges` |
| Train/Test split | 80% / 20%, `stratify=y`, `random_state=42` |
| Data leakage prevention | Preprocessor fitted **only on training data** |

**Shapes after preprocessing:**

```
Training raw:       (5634, 17)
Training processed: (5634, 39)   ← after OHE expansion
```

---

## 🤖 Model Training & Comparison

Three models were trained with **balanced class weights** to handle class imbalance:

| Model | Config |
|---|---|
| Logistic Regression | `max_iter=1000`, `class_weight='balanced'` |
| Random Forest | `n_estimators=100`, `max_depth=10`, `class_weight='balanced'` |
| XGBoost | `n_estimators=100`, `learning_rate=0.1`, `max_depth=5` |

### Baseline Test Set Results

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| **Logistic Regression** | 73.7% | 50.3% | **78.3%** | 61.2% | **84.2%** |
| Random Forest | 75.5% | 52.7% | 74.6% | 61.8% | 83.7% |
| XGBoost | **79.6%** | **64.0%** | 52.7% | 57.8% | 83.9% |

### Classification Reports (Test Set)

**Logistic Regression**
```
              precision  recall  f1-score  support
No  (0)         0.90     0.72     0.80     1035
Yes (1)         0.50     0.78     0.61      374
accuracy                          0.74     1409
```

**Random Forest**
```
              precision  recall  f1-score  support
No  (0)         0.89     0.76     0.82     1035
Yes (1)         0.53     0.75     0.62      374
accuracy                          0.76     1409
```

**XGBoost**
```
              precision  recall  f1-score  support
No  (0)         0.84     0.89     0.87     1035
Yes (1)         0.64     0.53     0.58      374
accuracy                          0.80     1409
```

### Overfitting Check (Train vs Test Recall)

| Model | Train Recall | Test Recall | Assessment |
|---|---|---|---|
| Logistic Regression | 80.9% | 78.3% | ✅ Minimal gap |
| Random Forest | **96.6%** | 74.6% | ⚠️ Overfitting |
| XGBoost | — | 52.7% | ⚠️ Under-detects churners |

> **Selected Model: Logistic Regression** — best recall with no overfitting, directly aligned with the business goal of maximising churn detection.

---

## ⚙️ Hyperparameter Tuning — Logistic Regression

`GridSearchCV` with **recall** as the scoring metric was run to find optimal parameters.

**Best Parameters Found:**
```python
{
  'C': 0.01,
  'class_weight': 'balanced',
  'penalty': 'l1',
  'solver': 'liblinear'
}
```

**Best Cross-Validation Recall:** `82.27%`

### Tuned Model — Final Test Set Performance

| Metric | Score |
|---|---|
| **Accuracy** | 72.4% |
| **Precision** | 48.8% |
| **Recall** | **82.1%** ✅ |
| **F1-Score** | 61.2% |
| **ROC-AUC** | **83.1%** ✅ |

**Classification Report:**
```
              precision  recall  f1-score  support
No  (0)         0.91     0.69     0.79     1035
Yes (1)         0.49     0.82     0.61      374
accuracy                          0.72     1409
```

> Recall improved from **78.3% → 82.1%** vs baseline. The strong L1 penalty (`C=0.01`) drives many feature coefficients to exactly zero, acting as automatic feature selection.

---

## 🏆 Feature Importance (L1 Coefficients)

Only **7 out of 39 encoded features** received non-zero coefficients after L1 regularization:

| Rank | Feature | Coefficient | Effect on Churn |
|---|---|---|---|
| 1 | `contract` | −0.8539 | Longer contract → **reduces** churn |
| 2 | `tenure` | −0.5533 | More months → **reduces** churn |
| 3 | `monthlycharges` | +0.5270 | Higher charges → **increases** churn |
| 4 | `onlinesecurity = No` | +0.2169 | No security → **increases** churn |
| 5 | `paymentmethod = Electronic check` | +0.1693 | E-check → **increases** churn |
| 6 | `paperlessbilling = No` | −0.1341 | No paperless → **reduces** churn |
| 7 | `techsupport = No` | +0.1088 | No support → **increases** churn |

**Zeroed-out features** (L1 eliminated — no predictive value): `partner`, `dependents`, `multipleLines`, `internetService`, `onlineBackup`, `deviceProtection`, `streamingTV`, `streamingMovies`, `totalCharges`, `seniorcitizen`

---

## 💡 Final Business Insights

### Key Churn Drivers

1. **Contract Type** — Month-to-month customers have the highest churn risk. Moving them to longer contracts is the single most impactful retention lever.
2. **Tenure** — New customers are the most vulnerable. Strong onboarding programs are essential.
3. **Monthly Charges** — High-bill customers churn more. Pricing review needed for at-risk segments.
4. **Online Security** — Customers without it are more likely to churn. Promoting this add-on helps retention.
5. **Payment Method** — Electronic check users show the highest churn risk.
6. **Tech Support** — Customers without tech support churn slightly more.

### Business Recommendations

| Priority | Action |
|---|---|
| 🔴 High | Offer incentives to move month-to-month customers to 1- or 2-year contracts |
| 🔴 High | Create tailored onboarding programs for new (low-tenure) customers |
| 🟡 Medium | Review and restructure pricing for high monthly-charge customers |
| 🟡 Medium | Promote online security and tech support add-ons to at-risk segments |
| 🟢 Low | Investigate the higher churn rate associated with electronic check payments |

---

## 🌐 API Reference

### `POST /predict`

**Request Body (7 fields — only non-zero importance features):**
```json
{
  "tenure": 5,
  "onlinesecurity": "No",
  "techsupport": "No",
  "contract": "Month-to-month",
  "paperlessbilling": "Yes",
  "paymentmethod": "Electronic check",
  "monthlycharges": 70.0
}
```

**Response:**
```json
{
  "churn_probability": 0.769,
  "prediction": 1,
  "churn": "Yes",
  "risk_level": "HIGH",
  "model_name": "Logistic Regression",
  "model_version": "1.0"
}
```

### Risk Level Thresholds

| Level | Probability |
|---|---|
| 🔴 HIGH | ≥ 70% |
| 🟡 MEDIUM | 40% – 69% |
| 🟢 LOW | < 40% |

### Other Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | API status |
| GET | `/health` | Model health check |
| GET | `/docs` | Swagger UI (auto-generated) |
