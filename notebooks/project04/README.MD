# Titanic Fare Prediction: Regression Model Comparison

This project analyzes how different regression models perform when predicting passenger fares in the Titanic dataset. The goal is to compare simple linear models, regularized models, and polynomial models, and to understand which features carry meaningful predictive power.

## Project Workflow

### 1. Initial Models (Cases 1–4)
The project began with four simple feature sets:
- `age`
- `family_size`
- `sex`
- `age + sex`

These models were intentionally simple but performed poorly, showing very low R² values and high error metrics.

### 2. Expanded Feature Sets (Cases 5–6)
Because Cases 1–4 were weak, two stronger feature sets were added:
- `pclass`
- `pclass + sex`

These models performed significantly better and became the basis for the remaining work.

### 3. Regularized Models
To improve generalization and reduce overfitting:
- **Ridge Regression** was applied using `pclass + sex`
- **Elastic Net** was then applied to the same feature set

Elastic Net provided the best regularized performance.

### 4. Polynomial Regression
Polynomial transformations were explored in two ways:
- **Cubic and 8th-degree fits** using `age` (for visualization of nonlinear curves)
- **Polynomial Regression on `pclass + sex`**, which became the top-performing model overall

### 5. Model Comparison
All models were evaluated using:
- R²  
- RMSE  
- MAE  

A summary table in the notebook highlights the best and worst performers.

## Key Findings
- Single-feature linear models (Cases 1–4) are extremely weak predictors.
- `pclass` is the strongest single predictor.
- Combining `pclass + sex` improves performance further.
- Elastic Net outperforms Ridge for the same feature set.
- Polynomial Regression (`pclass + sex`) achieves the best overall metrics.
- High-degree polynomial fits on `age` reveal curvature but do not meaningfully improve prediction accuracy.

## Tools Used
- Python  
- pandas  
- scikit-learn  
- matplotlib  
- Jupyter Notebook  

## Purpose
This project demonstrates the process of:
- testing multiple feature sets,
- applying different types of regression,
- identifying underfitting/overfitting,
- and selecting the most predictive model.

It emphasizes clear reasoning, exploration, and comparison rather than producing a perfect fare-prediction model.
