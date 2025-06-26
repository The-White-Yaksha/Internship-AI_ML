# 📊 Internship Developers Hub Corporation Tasks

This repository contains three machine learning and data analysis tasks performed during an internship. Each notebook demonstrates practical applications of Python, exploratory data analysis (EDA), and predictive modeling using real-world datasets.

---

## 🐧 Task 1: **Exploratory Data Analysis on Penguin Dataset**

### 🔍 Objective:

To perform visual exploration and feature analysis on the Palmer Penguins dataset using Seaborn.

### 📁 Dataset:

* **Source:** `sns.load_dataset('penguins')`
* **Features:** Species, island, bill length/depth, flipper length, body mass, sex

### 🧪 Steps Performed:

1. **Data Loading & Initial Inspection**

   * Used `sns.load_dataset()` to fetch the dataset.
   * Inspected with `.head()` and `.info()`.

2. **Handling Missing Values**

   * Identified nulls and removed them for clean analysis.

3. **Data Visualization**

   * **Pairplot**: Visualized multi-feature relationships grouped by species.
   * **Violin Plot**: Compared bill length by island and species.
   * **Count Plot**: Displayed count of male/female penguins per species.
   * **Scatter Plot**: Analyzed bill depth vs. flipper length.

### 📊 Libraries Used:

```python
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
```

### ✅ Outcome:

This EDA revealed how species and island locations influence various physical attributes of penguins. It also highlighted the effectiveness of visualizations in distinguishing species based on measurements.

---

## ❤️ Task 3: **Heart Disease Prediction Using Logistic Regression**

### 🔍 Objective:

To build a binary classifier that predicts whether a person is at risk of heart disease.

### 📁 Dataset:

* **CSV Used:** `heart-disease.csv`
* **Target Variable:** Presence of Heart Disease (`1` = Yes, `0` = No)

### 🧪 Steps Performed:

1. **Data Loading & Cleaning**

   * Loaded dataset using Pandas.
   * Removed the `sex` column to avoid gender bias.

2. **Exploratory Analysis**

   * Used `.info()`, `.describe()`, and basic charts to understand data distribution.
   * Checked for null values and types.

3. **Model Building**

   * Split features and labels.
   * Applied `LogisticRegression()` from `sklearn.linear_model`.

4. **Model Evaluation**

   * Accuracy: \~90%
   * Model performed well with clean data and minimal preprocessing.

### 📊 Libraries Used:

```python
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
```

### ✅ Outcome:

Achieved strong results with logistic regression due to the cleanliness and simplicity of the dataset. Also demonstrated good feature selection and basic preprocessing.

---

## 🏠 Task 6: **House Price Prediction Using Advanced Regression Techniques**

### 🔍 Objective:

To build and evaluate predictive models to estimate Seattle house prices.

### 📁 Dataset:

* `train.csv` and `test.csv` from Kaggle
* **Target Variable:** `price`

### 🧪 Steps Performed:

#### 1. Data Cleaning & Preparation

* Identified and removed:

  * Null values
  * Duplicate entries
  * Irrelevant columns (e.g., `zip_code`, `size`, `country`)
* Unified units (converted acre-based sizes to square feet where necessary)

#### 2. Exploratory Data Analysis (EDA)

* Used histograms and correlation matrices
* Found outliers and strong predictors (like `sqft_living`, `bedrooms`, `bathrooms`, etc.)

#### 3. Regression Modeling

* **Linear Regression:**

  * Low R² score due to underfitting

* **Polynomial Regression (degree=2):**

  * Transformed features and saw improvement

* **Normalization:**

  * Scaled data using MinMaxScaler and StandardScaler for better performance

* **Ridge & Lasso Regression:**

  * Improved generalization
  * Ridge: Balanced performance
  * Lasso: Feature selection through penalization

#### 4. Evaluation Metrics:

* **R² Score**
* **MAE** (Mean Absolute Error)
* **MSE** (Mean Squared Error)
* **RMSE** (Root Mean Squared Error)

### 📊 Libraries Used:

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
```

### ✅ Outcome:

This task demonstrated your ability to iterate over models, clean data effectively, and evaluate using multiple metrics. Transitioning from basic linear regression to polynomial and regularized models greatly improved performance.

---

## 🧠 Overall Learnings

* Advanced data cleaning & preprocessing techniques
* Strong understanding of regression/classification fundamentals
* Application of regularization methods (Ridge, Lasso)
* Feature engineering and transformation (Polynomial features)
* Metric-driven evaluation and model comparison

---

## 📁 File Structure

```
.
├── Internship Task 1.ipynb      # Penguin EDA
├── Internship Task 3.ipynb      # Heart Disease Prediction
├── Internship Task 6.ipynb      # House Price Prediction
└── README.md                    # This Report
