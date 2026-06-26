#  Lending Club Loan Data — Data Understanding & EDA

## Project Objective
Build a **supervised machine learning model** to predict whether a borrower will **fully pay off their loan** or default, using historical Lending Club data.

---

## Datasets Used
| File | Description |
|------|-------------|
| `lending_club_loan_two.csv` | Main loan dataset with borrower & loan features |
| `lending_club_info.csv` | Metadata file describing each column/feature |

---

### 1. Library Imports & Data Loading
- Imported `pandas`, `seaborn`, and `matplotlib` for data manipulation and visualization.
- Loaded both the main loan dataset and the metadata file.
- Printed shapes of both DataFrames to understand the size of the data.

---

### 2. Initial Data Inspection
- Used `df.head()` and `df.head().T` (transposed) to preview the first few rows and get a better column-wise view of the data.
- Referenced `meta_data["Description"]` to understand what individual columns mean — important for domain understanding before analysis.
- Used `df.info()` to check column data types, non-null counts, and memory usage.

---

### 3. Target Variable Analysis — `loan_status`
- Plotted a **countplot** of `loan_status` to visualize class distribution.
- Used `value_counts()` to get exact counts.
- **Key Observation:** The data is **imbalanced** — "Fully Paid" loans are significantly more than "Charged Off" (defaulted). Training a model directly on this would result in a biased model.

---

### 4. Correlation Heatmap
- Computed a **correlation matrix** for all numeric columns and visualized it using a heatmap.
- Helps identify **multicollinear features** (e.g., features that are strongly correlated with each other, which may need to be removed before modeling).

---

### 5. Loan Amount Distribution & Relationships
- Plotted a **histogram** of `loan_amnt` to understand its distribution.
- Created a **scatter plot** of `loan_amnt` vs `installment` — these two are expected to be highly correlated since installment is derived from loan amount.
- Plotted `total_acc` vs `open_acc` to examine the relationship between total credit accounts and currently open ones.

---

### 6. Grade & Sub-Grade Analysis
- Explored `grade` and `sub_grade` (Lending Club's internal risk rating).
- Plotted **countplots with `loan_status` as hue** to see how default rates vary across risk grades — higher grades (E, F, G) are expected to have more defaults.

---

### 7. Loan Term Analysis
- Checked distribution of `term` (36-month vs 60-month loans).
- Plotted countplot with `loan_status` hue — longer-term loans may carry higher default risk.

---

### 8. Loan Amount by Loan Status
- Grouped by `loan_status` and described `loan_amnt` to compare statistics between fully paid and charged-off loans.
- Created a **box plot** to visualize distribution differences.
- Identified **outliers** in `loan_amnt` using the **IQR method** (1.5 × IQR rule).

---

### 9. Interest Rate Analysis
- Used `describe()` on `int_rate` to get summary statistics (min, max, mean, std).
- Higher interest rates are typically assigned to riskier borrowers.

---

### 10. Employment Features
- Explored `emp_title` (job title) — high cardinality column with many unique values.
- Explored `emp_length` (years of employment) — plotted countplot with `loan_status` hue to check if employment duration correlates with repayment behavior.

---

### 11. Home Ownership Analysis
- Explored `home_ownership` categories (RENT, MORTGAGE, OWN, etc.).
- Plotted countplot with `loan_status` hue — home ownership can be an indicator of financial stability.

---

### 12. Annual Income Analysis
- Plotted **histogram** of `annual_inc` to understand income distribution.
- Grouped by `loan_status` and described `annual_inc` — higher-income borrowers may be less likely to default.

---

### 13. Verification Status
- Plotted `verification_status` (Verified / Source Verified / Not Verified) against `loan_status`.
- Interesting to check whether income verification correlates with loan repayment.

---

### 14. Purpose of Loan
- Used `value_counts()` on `purpose` to see why borrowers are taking loans (debt consolidation, credit card, home improvement, etc.).

---

### 15. Debt-to-Income Ratio (DTI)
- Used `describe()` on `dti` column for summary statistics.
- Plotted a **box plot** of `dti` by `loan_status` — higher DTI generally indicates higher financial stress and potentially higher default risk.

---

## Key Findings

| Finding | Implication |
|---------|-------------|
| **Class Imbalance** in `loan_status` | Need to handle with oversampling/undersampling before modeling |
| **`loan_amnt` & `installment` are highly correlated** | May need to drop one to avoid multicollinearity |
| **Higher grade (E/F/G) → more defaults** | `grade`/`sub_grade` are strong predictors |
| **Outliers present in `loan_amnt`** | Need outlier treatment in preprocessing |
| **`emp_title` has very high cardinality** | May need to be dropped or encoded carefully |
| **Higher DTI → more charge-offs** | `dti` is a useful predictor |

---

## Tech Stack
- **Python 3**
- `pandas` — data manipulation
- `seaborn` — statistical visualizations
- `matplotlib` — plotting

---

## Next Steps
1. Data Preprocessing (handling nulls, encoding categoricals)
2. Address class imbalance (SMOTE / class weights)
3. Feature Engineering & Selection
4. Model Building (Logistic Regression, Random Forest, XGBoost, etc.)
5. Model Evaluation (ROC-AUC, Precision-Recall)