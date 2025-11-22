# **Wine Quality Prediction: Ensemble Model Comparison Project**
**Author:** Joanna Farris  
**Date:** November 20, 2025  
**Objective:** Explore whether ensemble methods (AdaBoost, Random Forest, and a Voting Classifier) can improve classification performance on the Wine dataset compared to the individual base models.

## **Introduction**

This project explores different machine learning models to predict outcomes from our dataset. I will test Random Forest, Decision Tree, SVM, Neural Network, and a Voting classifier to evaluate their performance. The goal is to identify which approaches work best, understand why, and explore ways to improve predictive accuracy.

#### **Imports**


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
    BaggingClassifier,
    VotingClassifier,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

```

#### **Section 1. Load and Inspect the Data**


```python
# Load the dataset (download from UCI and save in the same folder)
df = pd.read_csv("winequality-red.csv", sep=";")

# Display structure and first few rows
df.info()
df.head()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1599 entries, 0 to 1598
    Data columns (total 12 columns):
     #   Column                Non-Null Count  Dtype  
    ---  ------                --------------  -----  
     0   fixed acidity         1599 non-null   float64
     1   volatile acidity      1599 non-null   float64
     2   citric acid           1599 non-null   float64
     3   residual sugar        1599 non-null   float64
     4   chlorides             1599 non-null   float64
     5   free sulfur dioxide   1599 non-null   float64
     6   total sulfur dioxide  1599 non-null   float64
     7   density               1599 non-null   float64
     8   pH                    1599 non-null   float64
     9   sulphates             1599 non-null   float64
     10  alcohol               1599 non-null   float64
     11  quality               1599 non-null   int64  
    dtypes: float64(11), int64(1)
    memory usage: 150.0 KB





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>fixed acidity</th>
      <th>volatile acidity</th>
      <th>citric acid</th>
      <th>residual sugar</th>
      <th>chlorides</th>
      <th>free sulfur dioxide</th>
      <th>total sulfur dioxide</th>
      <th>density</th>
      <th>pH</th>
      <th>sulphates</th>
      <th>alcohol</th>
      <th>quality</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>7.4</td>
      <td>0.70</td>
      <td>0.00</td>
      <td>1.9</td>
      <td>0.076</td>
      <td>11.0</td>
      <td>34.0</td>
      <td>0.9978</td>
      <td>3.51</td>
      <td>0.56</td>
      <td>9.4</td>
      <td>5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>7.8</td>
      <td>0.88</td>
      <td>0.00</td>
      <td>2.6</td>
      <td>0.098</td>
      <td>25.0</td>
      <td>67.0</td>
      <td>0.9968</td>
      <td>3.20</td>
      <td>0.68</td>
      <td>9.8</td>
      <td>5</td>
    </tr>
    <tr>
      <th>2</th>
      <td>7.8</td>
      <td>0.76</td>
      <td>0.04</td>
      <td>2.3</td>
      <td>0.092</td>
      <td>15.0</td>
      <td>54.0</td>
      <td>0.9970</td>
      <td>3.26</td>
      <td>0.65</td>
      <td>9.8</td>
      <td>5</td>
    </tr>
    <tr>
      <th>3</th>
      <td>11.2</td>
      <td>0.28</td>
      <td>0.56</td>
      <td>1.9</td>
      <td>0.075</td>
      <td>17.0</td>
      <td>60.0</td>
      <td>0.9980</td>
      <td>3.16</td>
      <td>0.58</td>
      <td>9.8</td>
      <td>6</td>
    </tr>
    <tr>
      <th>4</th>
      <td>7.4</td>
      <td>0.70</td>
      <td>0.00</td>
      <td>1.9</td>
      <td>0.076</td>
      <td>11.0</td>
      <td>34.0</td>
      <td>0.9978</td>
      <td>3.51</td>
      <td>0.56</td>
      <td>9.4</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>



## **Section 2. Prepare the Data**

Step 1: Create a categorical label from the numeric target


```python
def quality_to_label(q):
    if q <= 4:
        return "low"
    elif q <= 6:
        return "medium"
    else:
        return "high"

# Call the apply() method on the quality column to create the new quality_label column
df["quality_label"] = df["quality"].apply(quality_to_label)
```

Step 2: Create a numeric label for modeling


```python
def quality_to_number(q):
    if q <= 4:
        return 0
    elif q <= 6:
        return 1
    else:
        return 2

# Call the apply() method on the quality column to create the new quality_numeric column
df["quality_numeric"] = df["quality"].apply(quality_to_number)
```

After these two steps, your dataframe has two new columns: 

| Column| Type | Meaning |
|:---|:---|:---|
|quality_label|string|Categorical label: "low", "medium", "high"|
|quality_numeric|integer|Numeric label: 0 = low, 1 = medium, 2 = high|

  

## **Section 3. Feature Selection and Justification**


```python
# Define input features (X)
X = df.drop(columns=["quality", "quality_label", "quality_numeric"])

# Define target variable
y = df["quality_numeric"]
```

✅ Justification:  

- **Features (X)**: 11 physicochemical properties of wine → meaningful predictors for wine quality.
- **Target (y)**: numeric category of wine quality → makes the classification problem manageable and compatible with sklearn models.

## **Section 4. Split the Data into Train and Test**


```python
# Train/test split (stratify to preserve class balance)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

#### **Helper function to train and evaluate models** 


```python
# Helper function
def evaluate_model(name, model, X_train, y_train, X_test, y_test, results):
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    train_f1 = f1_score(y_train, y_train_pred, average="weighted")
    test_f1 = f1_score(y_test, y_test_pred, average="weighted")

    print(f"\n{name} Results")
    print("Confusion Matrix (Test):")
    print(confusion_matrix(y_test, y_test_pred))
    print(f"Train Accuracy: {train_acc:.4f}, Test Accuracy: {test_acc:.4f}")
    print(f"Train F1 Score: {train_f1:.4f}, Test F1 Score: {test_f1:.4f}")

    results.append(
        {
            "Model": name,
            "Train Accuracy": train_acc,
            "Test Accuracy": test_acc,
            "Train F1": train_f1,
            "Test F1": test_f1,
        }
    )

```


```python
results = []
```

#### **Model #1: Random Forest (100 trees)**


```python
# Define Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train and evaluate using the same helper function
evaluate_model("Random Forest (100)", rf_model, X_train, y_train, X_test, y_test, results)

# Check updated results
results
```

    
    Random Forest (100) Results
    Confusion Matrix (Test):
    [[  0  13   0]
     [  0 256   8]
     [  0  15  28]]
    Train Accuracy: 1.0000, Test Accuracy: 0.8875
    Train F1 Score: 1.0000, Test F1 Score: 0.8661





    [{'Model': 'Random Forest (100)',
      'Train Accuracy': 1.0,
      'Test Accuracy': 0.8875,
      'Train F1': 1.0,
      'Test F1': 0.8660560842649911}]



#### **Model #2: Voting (DT + SVM + NN)**


```python
# 1. Scale the features once
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 2. Define base models
dt_model = DecisionTreeClassifier()
svc_model = SVC(probability=True)  # Required for soft voting
nn_model = MLPClassifier(hidden_layer_sizes=(50,), solver='lbfgs', max_iter=1000)

# 3. Create voting ensemble
voting_model = VotingClassifier(
    estimators=[('DT', dt_model), ('SVM', svc_model), ('NN', nn_model)],
    voting='soft'
)

# 4. Train and evaluate models using scaled data
# Use the same scaled data for all models
evaluate_model("Decision Tree", dt_model, X_train_scaled, y_train, X_test_scaled, y_test, results)
evaluate_model("SVM", svc_model, X_train_scaled, y_train, X_test_scaled, y_test, results)
evaluate_model("Neural Net", nn_model, X_train_scaled, y_train, X_test_scaled, y_test, results)
evaluate_model("Voting (DT + SVM + NN)", voting_model, X_train_scaled, y_train, X_test_scaled, y_test, results)

# 5. Inspect the results
results

```

    
    Decision Tree Results
    Confusion Matrix (Test):
    [[  1  10   2]
     [  8 229  27]
     [  3  10  30]]
    Train Accuracy: 1.0000, Test Accuracy: 0.8125
    Train F1 Score: 1.0000, Test F1 Score: 0.8188
    
    SVM Results
    Confusion Matrix (Test):
    [[  0  13   0]
     [  0 254  10]
     [  0  25  18]]
    Train Accuracy: 0.8569, Test Accuracy: 0.8500
    Train F1 Score: 0.8204, Test F1 Score: 0.8219
    
    Neural Net Results
    Confusion Matrix (Test):
    [[  3   9   1]
     [  6 237  21]
     [  0  13  30]]
    Train Accuracy: 1.0000, Test Accuracy: 0.8438
    Train F1 Score: 1.0000, Test F1 Score: 0.8437
    
    Voting (DT + SVM + NN) Results
    Confusion Matrix (Test):
    [[  0  13   0]
     [  3 246  15]
     [  0  13  30]]
    Train Accuracy: 1.0000, Test Accuracy: 0.8625
    Train F1 Score: 1.0000, Test F1 Score: 0.8489





    [{'Model': 'Random Forest (100)',
      'Train Accuracy': 1.0,
      'Test Accuracy': 0.8875,
      'Train F1': 1.0,
      'Test F1': 0.8660560842649911},
     {'Model': 'Decision Tree',
      'Train Accuracy': 1.0,
      'Test Accuracy': 0.8125,
      'Train F1': 1.0,
      'Test F1': 0.8188438252493979},
     {'Model': 'SVM',
      'Train Accuracy': 0.8569194683346364,
      'Test Accuracy': 0.85,
      'Train F1': 0.8204335064299121,
      'Test F1': 0.8219107812341676},
     {'Model': 'Neural Net',
      'Train Accuracy': 1.0,
      'Test Accuracy': 0.84375,
      'Train F1': 1.0,
      'Test F1': 0.8436535114402555},
     {'Model': 'Voting (DT + SVM + NN)',
      'Train Accuracy': 1.0,
      'Test Accuracy': 0.8625,
      'Train F1': 1.0,
      'Test F1': 0.8488954375848033}]



## **Section 6. Compare Results** 


```python
# Create a table of results 
results_df = pd.DataFrame(results)

results_df["Accuracy Gap"] = results_df["Train Accuracy"] - results_df["Test Accuracy"]
results_df["F1 Gap"] = results_df["Train F1"] - results_df["Test F1"]

# Reorder and rename columns for readability
results_df = results_df[
    ["Model", "Train Accuracy", "Test Accuracy", "Accuracy Gap", "Train F1", "Test F1", "F1 Gap"]
]

# Sort by Test Accuracy
results_df.sort_values(by="Test Accuracy", ascending=False, inplace=True)

print("\nSummary of All Models: Ranked by Test Accuracy")
display(results_df)

```

    
    Summary of All Models: Ranked by Test Accuracy



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Model</th>
      <th>Train Accuracy</th>
      <th>Test Accuracy</th>
      <th>Accuracy Gap</th>
      <th>Train F1</th>
      <th>Test F1</th>
      <th>F1 Gap</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Random Forest (100)</td>
      <td>1.000000</td>
      <td>0.88750</td>
      <td>0.112500</td>
      <td>1.000000</td>
      <td>0.866056</td>
      <td>0.133944</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Voting (DT + SVM + NN)</td>
      <td>1.000000</td>
      <td>0.86250</td>
      <td>0.137500</td>
      <td>1.000000</td>
      <td>0.848895</td>
      <td>0.151105</td>
    </tr>
    <tr>
      <th>2</th>
      <td>SVM</td>
      <td>0.856919</td>
      <td>0.85000</td>
      <td>0.006919</td>
      <td>0.820434</td>
      <td>0.821911</td>
      <td>-0.001477</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Neural Net</td>
      <td>1.000000</td>
      <td>0.84375</td>
      <td>0.156250</td>
      <td>1.000000</td>
      <td>0.843654</td>
      <td>0.156346</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Decision Tree</td>
      <td>1.000000</td>
      <td>0.81250</td>
      <td>0.187500</td>
      <td>1.000000</td>
      <td>0.818844</td>
      <td>0.181156</td>
    </tr>
  </tbody>
</table>
</div>


## **Section 7. Conclusions and Insights**

Random Forest performed best, with 88.75% test accuracy and an F1 of 0.866. It likely beat a single decision tree because combining 100 trees reduces overfitting while capturing different patterns in the data. The Voting classifier improved on its base models, but nothing topped Random Forest. SVM was stable but slightly underfit, and the neural net overfit a bit. Next steps could be tuning Random Forest and adding some feature engineering to boost performance.
