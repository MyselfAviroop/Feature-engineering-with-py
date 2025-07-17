# 🧠 Data Preprocessing & Encoding Techniques in Machine Learning

This repository contains various data preprocessing techniques demonstrated using Python, pandas, seaborn, scikit-learn, and SMOTE. These techniques are essential for preparing data before feeding it into machine learning models.

---

## 📁 Contents

### 1. Titanic Dataset Exploration & Imputation
- **Dataset**: Loaded using Seaborn's `titanic` dataset.
- **Actions**:
  - Checked for missing values.
  - Dropped rows/columns with nulls.
  - Visualized age distribution.
  - Handled missing values using:
    - Mean Imputation
    - Median Imputation
    - Mode Imputation (for categorical `embarked` column)

---

### 2. SMOTE for Imbalanced Data
- **Created a synthetic classification dataset** using `make_classification`.
- Visualized class imbalance.
- Applied **SMOTE (Synthetic Minority Over-sampling Technique)** to balance classes.
- Visualized the balanced dataset.

---

### 3. Outlier Detection using Boxplot and IQR
- Used a list of student marks with intentional outliers.
- Calculated:
  - Q1, Q2 (Median), Q3
  - IQR (Interquartile Range)
  - Lower & Upper Fences
- Visualized the data using a **boxplot** to highlight outliers.

---

### 4. Categorical Encoding Techniques

#### 🔹 One-Hot Encoding
- Used `OneHotEncoder` from scikit-learn to convert categorical values into binary vectors.

#### 🔹 Label Encoding
- Used `LabelEncoder` to convert string labels into integer values.

#### 🔹 Ordinal Encoding
- Used `OrdinalEncoder` to encode ordered categories like `small < medium < large`.

#### 🔹 Target-Guided Ordinal Encoding
- Encoded the `city` column based on the mean of the `price` column.
- Useful when encoding categorical variables that correlate with the target.

---

## 🛠 Libraries Used

- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `sklearn`
- `imblearn` (for SMOTE)

---

## ✅ How to Run

1. Make sure you have Python installed (preferably 3.8+)
2. Install required libraries:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn
