# 📊 Lending Club Loan Default Prediction

Predicting whether a borrower will fully repay or default on a loan, using historical Lending Club data and gradient-boosted / ensemble tree models.

## 📖 About The Project

**Lending Club** was a peer-to-peer lending platform that connected individual borrowers with investors willing to fund personal loans. Each loan record contains borrower attributes (income, credit history, employment, home ownership, debt ratios, etc.) along with the final outcome of the loan.

**Loan default prediction** is the task of using a borrower's application-time information to predict whether they will *fully repay* the loan or *default* (charge off) before it matures.

This matters because:

- **Risk management** — lenders need to price risk correctly and decide which applicants to approve.
- **Capital allocation** — investors funding loans want to avoid capital tied up in loans that are unlikely to be repaid.
- **Early warning** — a reliable default classifier lets an institution flag high-risk applications *before* funds are disbursed rather than after a borrower has already missed payments.

This project was built to walk through a complete, realistic credit-risk modeling pipeline — from raw tabular loan data through EDA, cleaning, encoding, and outlier handling, to training and comparing tree-based classifiers.

**Dataset overview:** the raw dataset (`lending_club_loan_two.csv`) contains **396,030 loan records across 27 columns**, covering loan terms, borrower credit attributes, and the loan outcome. A companion file (`lending_club_info.csv`) provides plain-language descriptions for each column.

**Target variable:** `loan_status`, originally a two-class string field (`Fully Paid` / `Charged Off`), mapped to a binary target — `1` = Default / Charged Off, `0` = Fully Paid. The dataset is **imbalanced**, with fully-paid loans making up the large majority of records.

### Built With

- [Python](https://www.python.org/)
- [Pandas](https://pandas.pydata.org/)
- [NumPy](https://numpy.org/)
- [Scikit-learn](https://scikit-learn.org/) — preprocessing (`OneHotEncoder`, `OrdinalEncoder`, `MinMaxScaler`), `train_test_split`, `RandomizedSearchCV`, `RandomForestClassifier`, evaluation metrics
- [XGBoost](https://xgboost.readthedocs.io/) — `XGBClassifier`
- [Matplotlib](https://matplotlib.org/) & [Seaborn](https://seaborn.pydata.org/) — EDA visualizations
- [Joblib](https://joblib.readthedocs.io/) — serializing processed train/test splits between notebooks
- [Jupyter Notebook](https://jupyter.org/)

---

## 🔄 Project Workflow

```
Data Collection
      ↓
Data Understanding
      ↓
Exploratory Data Analysis (EDA)
      ↓
Data Cleaning
      ↓
Missing Value Treatment
      ↓
Feature Engineering
      ↓
Categorical Encoding
      ↓
Train-Test Split
      ↓
Outlier Removal (train set)
      ↓
Feature Scaling (MinMaxScaler)
      ↓
Handling Class Imbalance (scale_pos_weight)
      ↓
Model Training (XGBoost, Random Forest)
      ↓
Hyperparameter Tuning (RandomizedSearchCV)
      ↓
Model Evaluation
      ↓
Model Comparison
```

---

## 🔍 Exploratory Data Analysis

Performed in `data_understanding.ipynb`:

- **Target distribution:** `loan_status` is imbalanced — the majority of loans are `Fully Paid`, confirming the need to account for class imbalance during modeling rather than training naively.
- **Correlation analysis:** a heatmap of all numeric features against each other, plus a dedicated bar chart of each numeric feature's correlation with the binary target, was used to identify the strongest linear relationships with default risk.
- **Numerical distributions:** histograms for `loan_amnt`, `annual_inc`, `int_rate`, `dti`, `open_acc`, `total_acc`, `revol_util`, and `revol_bal`, several split by `loan_status` to compare distributions between repaid and defaulted loans.
- **Categorical analysis:** count plots of `grade`, `sub_grade`, `term`, `emp_length`, `home_ownership`, and `verification_status` against `loan_status` to see how default rates vary across categories.
- **Outlier analysis:** IQR-based outlier detection on `loan_amnt`, plus manual thresholding on `annual_inc`, `open_acc`, `total_acc`, `revol_util`, and `revol_bal` by inspecting how many records fall beyond visually identified cutoffs.
- **Key observations from the notebook:**
  - `loan_amnt` and `installment` are strongly related (as expected, since installment is derived from loan amount and term).
  - Higher `int_rate` is associated with a lower chance of the loan being fully paid.
  - Charge-off rates are **very similar across all `emp_length` categories**, which is why `emp_length` was later dropped as a predictive feature.
  - A small number of borrowers report unusually high `annual_inc` (≥ 250,000, and even ≥ 1,000,000), `open_acc` (> 40), `total_acc` (> 80), `revol_util` (> 120), and `revol_bal` (> 250,000), motivating the outlier thresholds applied during preprocessing.

---

## 🧹 Data Preprocessing

Performed in `data_preprocessing.ipynb`, applied to the raw 396,030-row, 27-column dataset:

| Step | What was done | Why |
|---|---|---|
| Drop `emp_title` | Column removed entirely | Extremely high cardinality (free-text job titles), not usable directly as a categorical feature |
| Drop `emp_length` | Column removed entirely | Charge-off rates were nearly identical across every employment-length bucket, so it carried little predictive signal |
| Drop `title` | Column removed entirely | Free-text description that largely duplicates the `purpose` column |
| `mort_acc` missing values | Filled with the column mean | Missingness was limited and mean imputation preserved the overall distribution |
| Remaining missing values | Rows dropped via `dropna()` | Affected columns had under 0.5% missing data — dropping was cheaper than imputing and had negligible impact on dataset size (811 rows and 3 columns removed in total) |
| Target encoding | `loan_status` mapped `Fully Paid → 0`, `Charged Off → 1` | Converts the target into a binary classification label |
| `term` cleanup | Extracted the integer number of months from the string (e.g. `"36 months"` → `36`) | Converts a text field into a usable numeric feature |
| Drop `sub_grade` | Column removed | Redundant with `grade`, which already captures the same credit-risk tier at coarser granularity |
| `home_ownership` cleanup | Rare categories `NONE` and `ANY` merged into `OTHER` | Reduces sparse/rare categories before one-hot encoding |
| Drop `issue_d` | Column removed | Loan issue date is only known *after* a loan is approved — including it would leak information not available at prediction time |
| `earliest_cr_line` | Parsed to datetime, then reduced to just the year | Captures credit history length in a simpler numeric form |
| Address → `zip` | Extracted the 5-digit zip code from the `address` field; original `address` column dropped | Zip code carries more standardized signal than a raw free-text address string |
| Encoding strategy | `initial_list_status` label-encoded (0/1); `grade` ordinal-encoded (`G`→`A` from worst to best); `home_ownership`, `verification_status`, `purpose`, `application_type`, and `zip` one-hot encoded | Matches encoding technique to each feature's type — binary, ordinal, and nominal categories are handled differently for correctness |
| One-hot encoding fit strategy | Encoder **fit on the training set only**, then applied to the test set (`drop='first'`, `handle_unknown='ignore'`) | Prevents test-set categories from leaking into the encoding scheme and avoids errors on unseen categories at inference time |
| `pub_rec` / `pub_rec_bankruptcies` | Binarized to 0 (no records) vs 1 (one or more records) | These fields are heavily skewed toward zero; binarizing reduces noise from rare high counts |
| Train/test split | 70/30 split via `train_test_split(test_size=0.3, random_state=42)` | Standard hold-out evaluation set with a fixed seed for reproducibility |
| Outlier removal | Applied **only to the training set**: `annual_inc ≤ 250000`, `dti ≤ 50`, `open_acc ≤ 40`, `total_acc ≤ 80`, `revol_util ≤ 120`, `revol_bal ≤ 250000` | Removes extreme values identified during EDA that could distort model training, while leaving the test set untouched for an honest evaluation |
| Feature scaling | `MinMaxScaler` fit on `X_train`, then applied to `X_test` | Puts all numeric features on a comparable [0, 1] scale |
| Persisting splits | Final `X_train`, `X_test`, `y_train`, `y_test` saved together via `joblib.dump()` | Decouples preprocessing from model training so models can be trained/re-trained without repeating the full pipeline |

**Final processed shapes:** `X_train: (273881, 45)`, `X_test: (118566, 45)`, `y_train: (273881,)`, `y_test: (118566,)`.

---

## 🤖 Machine Learning Models

Trained and compared in `model_training.ipynb`:

### XGBoost Classifier (`XGBClassifier`)
- **Description:** a gradient-boosted decision tree ensemble that builds trees sequentially, each correcting the errors of the previous ones.
- **Why selected:** strong out-of-the-box performance on structured/tabular data, native handling of class imbalance via `scale_pos_weight`, and efficient training via the `hist` tree method.
- **Advantages:** typically high predictive accuracy on tabular data, built-in regularization, handles feature interactions well.
- **Limitations:** more hyperparameters to tune than simpler models, longer training time when combined with extensive hyperparameter search.
- **Class imbalance handling:** `scale_pos_weight` set to the ratio of negative to positive class counts in the training data.
- **Hyperparameter tuning:** `RandomizedSearchCV` over `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`, and `min_child_weight`, using 3-fold cross-validation and randomized hyperparameter search, with `average_precision` / PR-AUC used as the primary model-selection metric.
- **Best parameters found:** `n_estimators=500`, `max_depth=6`, `learning_rate=0.05`, `subsample=0.8`, `colsample_bytree=0.8`, `min_child_weight=5` (Best CV ROC-AUC: **0.9078**).

### Random Forest Classifier (`RandomForestClassifier`)
- **Description:** an ensemble of independently trained decision trees whose predictions are averaged/voted, trained here with `n_estimators=100` (default hyperparameters, no tuning applied).
- **Why selected:** used as a baseline ensemble comparison against the tuned XGBoost model.
- **Advantages:** simple to train, resistant to overfitting on individual trees, requires little preprocessing.
- **Limitations:** with default settings it can overfit the training data noticeably (as seen in the results below), and it wasn't hyperparameter-tuned in this project.

### Model Comparison

| Model | ROC-AUC | PR-AUC |
|---|---:|---:|
| **Random Forest** | **0.889** | **0.762** |
| **XGBoost** | **0.908** | **0.783** |

> XGBoost outperformed Random Forest on both ROC-AUC and PR-AUC, so it was selected as the final model. PR-AUC was prioritized because Default is the minority positive class, while ROC-AUC was retained as a complementary measure of overall discrimination.

---

## 📏 Model Evaluation

Both models were evaluated on the held-out test set using:

- **Accuracy Score** — overall proportion of correct predictions; reported for context but not the primary metric given class imbalance.
- **Precision, Recall, F1-score** (via `classification_report`) — reported per class, important because in loan default prediction the cost of missing a defaulter (false negative on the "Charged Off" class) is typically much higher than a false positive.
- **Confusion Matrix** — visualized with `ConfusionMatrixDisplay` for both models, showing the counts of correctly/incorrectly classified `Default` vs `Fully-Paid` loans on the test set.
- **ROC-AUC / ROC Curve** — used as a complementary measure of overall discrimination between Default and Fully-Paid borrowers.
- **PR-AUC / Precision-Recall Curve** — prioritized for the imbalanced default-class problem and used to assess precision-recall performance across classification thresholds.

These metrics matter here specifically because loan default prediction is an **imbalanced classification problem** — a model can score a high raw accuracy just by predicting "Fully Paid" for nearly everyone, which is exactly why recall on the default class, and ROC-AUC, are tracked alongside accuracy rather than relying on accuracy alone.

---

## 📊 Final Test Results

The final XGBoost model was evaluated on the held-out test set.

| Metric | XGBoost Test Result |
|---|---:|
| Accuracy | **81.01%** |
| Default Precision | **50.95%** |
| Default Recall | **79.93%** |
| Default F1-score | **62.23%** |
| ROC-AUC | **0.908** |
| PR-AUC | **0.783** |


## ✨ Key Features

- End-to-end machine learning pipeline: raw CSV → EDA → cleaning → encoding → scaling → model training → evaluation
- Data leakage awareness (e.g. dropping `issue_d`, fitting encoders/scalers on training data only)
- Feature engineering from raw text/date fields (`term`, `earliest_cr_line`, `address` → `zip`)
- Mixed categorical encoding strategy (label, ordinal, and one-hot) matched to feature type
- EDA-driven outlier handling based on observed distributions
- Imbalanced classification handling via `scale_pos_weight`
- Hyperparameter tuning with `RandomizedSearchCV` and cross-validation
- Multi-model comparison (XGBoost vs. Random Forest) with a shared evaluation framework
- Reusable train/test artifact (`processed_data.pkl`) decoupling preprocessing from modeling

---

## 🔮 Future Improvements

- [ ] Improve domain-specific feature engineering for borrower risk
- [ ] Perform deeper false-positive / false-negative error analysis
- [ ] Further tune XGBoost using Average Precision as the optimization metric
- [ ] Experiment with LightGBM and CatBoost
- [ ] Compare alternative class-imbalance strategies
- [ ] Evaluate probability calibration
- [ ] Tune the classification threshold using a business cost matrix
- [ ] Add SHAP-based model explainability
- [ ] Add model monitoring and drift detection
- [ ] Build a Streamlit interface for default-risk prediction

## 📁 Repository Structure

```
├── data_understanding.ipynb      # EDA: distributions, correlations, class balance
├── data_preprocessing.ipynb      # Cleaning, feature engineering, encoding, scaling
├── model_training.ipynb          # Model training, tuning, and evaluation
├── lending_club_loan_two.csv     # Raw loan-level dataset
├── lending_club_info.csv         # Column name → description mapping
├── README.md
└── requirements.txt
```

---

## ⚙️ Installation

```bash
# Clone the repository
git clone https://github.com/Debasishsaha123/Loan-default-prediction-ML-Projects.git

# Move into the project directory
cd Loan-default-prediction-ML-Projects

# Create a virtual environment
python -m venv venv

# Activate the environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter Notebook
jupyter notebook
```

Then open the notebooks in order: `data_understanding.ipynb` → `data_preprocessing.ipynb` → `model_training.ipynb`.

---

## 📝 License

This project is licensed under the MIT License.
