# Ames Housing Price Prediction

## Project Overview

This project focuses on predicting residential house prices using the Ames Housing Dataset from the Kaggle House Prices Competition.

The objective was not only to build an accurate machine learning model but also to perform a complete real-world machine learning workflow including:

* Data Cleaning
* Missing Value Imputation
* Outlier Detection
* Feature Engineering
* Feature Encoding
* Model Training
* Cross Validation
* Model Comparison
* Kaggle Submission

The project was completed as part of an intermediate Machine Learning Training Program assignment.

---

## Dataset

**Competition:** House Prices - Advanced Regression Techniques

The dataset contains:

* 1,460 training observations
* 1,459 test observations
* 79 explanatory variables
* Target Variable: `SalePrice`

The dataset includes numerical, categorical, and ordinal features describing various aspects of residential homes in Ames, Iowa.

---

## Project Workflow

### 1. Data Understanding & Exploration

* Dataset inspection
* Data type analysis
* Missing value analysis
* Distribution analysis
* Categorical feature inspection

### 2. Data Cleaning

* Identified structural missing values
* Imputed categorical and numerical features
* Removed extreme outliers from `GrLivArea`
* Eliminated unexplained null values

### 3. Target Transformation

Since `SalePrice` was highly right-skewed:

```python
SalePrice = np.log1p(SalePrice)
```

This significantly improved normality and model performance.

### 4. Feature Engineering

Created several new features:

| Feature      | Description                   |
| ------------ | ----------------------------- |
| TotalSF      | Total living area             |
| HouseAge     | Age of house at sale          |
| RemodelAge   | Years since remodeling        |
| TotalBath    | Combined bathroom score       |
| HasPool      | Binary pool indicator         |
| HasGarage    | Binary garage indicator       |
| HasFireplace | Binary fireplace indicator    |
| Has2ndFloor  | Binary second floor indicator |

### 5. Encoding

* Ordinal encoding for quality-based features
* One-hot encoding for nominal categorical features
* Train/Test column alignment

---

## Models Trained

### Linear Regression

Baseline model used for comparison.

### Ridge Regression

Regularized linear regression with cross-validated alpha selection.

### Lasso Regression

Feature selection and regularization using L1 penalty.

### Random Forest Regressor

Ensemble tree-based model used for capturing nonlinear relationships.

### Gradient Boosting Regressor

Best-performing model among all tested algorithms.

---

## Evaluation Metric

The competition uses:

**Root Mean Squared Error (RMSE)**

evaluated on:

```python
log1p(SalePrice)
```

5-Fold Cross Validation was used for reliable model evaluation.

---

## Visualizations

The project includes:

* Missing Value Analysis
* SalePrice Distribution
* Correlation Heatmap
* Feature Importance Plot
* Predicted vs Actual Plot
* Residual Analysis

---

## Project Structure

```text
Ames-Housing-Price-Prediction/
│
├── Ames_Housing_Assignment.ipynb
├── clean_train.csv
├── clean_test.csv
├── submission.csv
├── kaggle_score.png
├── README.md
```

---

## Key Learnings

* Handling real-world missing values
* Preventing data leakage
* Feature engineering for tabular data
* Model selection and validation
* Bias-variance tradeoff
* Cross-validation best practices
* Kaggle competition workflow

---

## Results

| Model                       | CV RMSE |
| --------------------------- | ------- |
| Linear Regression           | 0.148   |
| Ridge Regression            | 0.138   |
| Lasso Regression            | 0.134   |
| Random Forest Regressor     | 0.142   |
| Gradient Boosting Regressor | 0.128   |

**Best Model:** Gradient Boosting Regressor

**Target Achievement:** Successfully achieved the assignment target of CV RMSE < 0.14 on the log-transformed SalePrice target.

**Key Observation:** Regularized linear models (Ridge and Lasso) outperformed the baseline Linear Regression, while Gradient Boosting delivered the strongest generalization performance and the lowest cross-validation error.



---

## Kaggle Submission

The final model was used to generate predictions on the competition test set and submitted to Kaggle.


![Kaggle Score](Kaggle_score.png)

---

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* SciPy
* Scikit-Learn
* Google Colab
* GitHub
* Kaggle

---

## Author

**Kanishk**

Machine Learning Enthusiast | Data Science Learner

GitHub: https://github.com/FANGER7
