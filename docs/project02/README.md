# # Titanic Data Analysis – Project 2  
**Author:** Joanna Farris  
**Date:** October 29, 2025

---

You can view the full Jupyter notebook for this project [here](https://github.com/Pojetta/applied-ml-farris/blob/main/notebooks/project02/ml02_farris.ipynb)


## Overview  
This project applies exploratory data analysis and preprocessing techniques to the classic **Titanic dataset**, focusing on identifying the factors most strongly associated with passenger survival. The notebook demonstrates a complete workflow from inspection and cleaning to feature engineering and data splitting.  

---

## Dataset  
The Titanic dataset is loaded directly from the **Seaborn** library for consistency and simplicity.

It contains both categorical and numeric attributes describing passenger demographics, socioeconomic status, and survival outcomes.  

**Example features include:**  
- `age` – Passenger age  
- `fare` – Ticket fare  
- `pclass` – Passenger class (1st, 2nd, 3rd)  
- `sex` – Gender  
- `sibsp`, `parch` – Number of siblings/spouses and parents/children aboard  
- `survived` – Target variable (1 = survived, 0 = did not survive)  

---

## Methods  

### 1. Import and Inspect the Data  
- Loaded the Titanic dataset from Seaborn.  
- Reviewed structure, data types, and missing values.  
- Displayed summary statistics and numeric correlations.  

### 2. Data Exploration and Preparation  

#### 2.1 Explore Data Patterns and Distributions  
- Created scatter plots, histograms, and count plots to visualize relationships between features (e.g., age, fare, and class).  
- Used custom color palettes for visual clarity.  

#### 2.2 Handle Missing Values and Clean Data  
- Imputed missing `age` values with the median.  
- Filled missing `embark_town` values with the mode.  

#### 2.3 Feature Engineering  
- Created a new `family_size` feature (`sibsp + parch + 1`).  
- Encoded categorical variables (`sex`, `embarked`) numerically.  
- Converted binary features to integers for consistency.  

### 3. Feature Selection and Justification  
- Selected input features: `age`, `fare`, `pclass`, `sex`, `family_size`.  
- Chose `survived` as the target variable for classification.  

### 4. Splitting  

#### 4.1 Basic Train/Test Split  
- Used `train_test_split` to divide the data into training and testing sets.  

#### 4.2 Stratified Train/Test Split  
- Applied `StratifiedShuffleSplit` to preserve the same survival ratio in both training and test sets.  

#### 4.3 Compare Survival Ratios  
- Compared survival proportions across the original, training, and testing sets.  
- Observed that the **stratified split** maintained the original ratio (≈61.6% non-survivors, 38.4% survivors),  
  while the **basic split** showed minor deviation between subsets.  



---

## Results  
- The dataset required moderate cleaning, mainly addressing missing `age` and `embark_town` values.  
- Visualizations showed clear patterns between **fare**, **class**, and **survival**.  
- The engineered `family_size` feature provides a more holistic measure of group travel dynamics.  
- Stratified splitting maintained consistent survival proportions across train and test sets, ensuring fairer model evaluation in future steps.  

---

## Tools and Libraries  
- **Python 3.12**  
- **Pandas**, **NumPy** – data handling and calculations  
- **Seaborn**, **Matplotlib** – data visualization  
- **scikit-learn** – data splitting and preparation  

---

## Reflection  
This project reinforced the importance of thoughtful preprocessing — especially handling missing values, encoding categorical data, and maintaining class balance when splitting data for modeling. These steps form the foundation for reliable machine learning workflows.  

