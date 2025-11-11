# Project 03
# Titanic Classification Project

This project uses the Titanic dataset to explore how different machine learning models perform when predicting passenger survival. The goal was to compare several models—Decision Tree, Support Vector Machine (SVM), and a Neural Network—using three simple feature sets and evaluate how well each model handled the task.

The notebook walks through data cleaning, feature selection, model training, evaluation, and visualizations.

---

## Project Structure (Notebook Outline)

### **1. Import and Inspect the Data**
Load the dataset, view the first rows, and confirm data types. This establishes the structure of the dataset before any transformations.

### **2. Data Exploration and Preparation**
#### 2.1 Handle Missing Values and Clean Data  
Address missing ages and other NaN values.  
#### 2.2 Feature Engineering  
Create new variables such as *family_size* and reduce others into usable formats.

---

## **3. Feature Selection**
#### 3.1 Choose Features and Target  
Three feature sets were tested:
- **Case 1:** alone  
- **Case 2:** age  
- **Case 3:** age + family_size  
Target variable: **survived**

#### 3.2 Define X and y  
Each feature set was turned into an X DataFrame and paired with the corresponding y Series, with missing values removed where necessary.

---

## **4. Decision Tree Model**
#### 4.1 Split the Data  
StratifiedShuffleSplit was used to preserve the survival ratio in both training and test sets.

#### 4.2 Create and Train Model  
A DecisionTreeClassifier was trained for each case without custom hyperparameters.

#### 4.3 Predict and Evaluate  
Classification reports were generated for both training and test sets.

#### 4.4 Confusion Matrices  
Heatmaps were plotted to visualize prediction patterns for each case.

#### 4.5 Decision Tree Plots  
Tree diagrams were generated and saved to show the learned splits.

---

## **5. Alternative Models (SVM and Neural Network)**

### **5.1 SVM Models**
SVM models were trained for each feature set using the default RBF kernel.  
Performance was evaluated on the test set.

### **5.2 Visualizing Support Vectors**
Support vectors were plotted for Case 1 (1D) and Case 3 (2D).  
These plots helped show how the SVM attempted to separate the classes based on the chosen features.

### **5.3 Neural Network**
A Multi-Layer Perceptron (MLP) classifier was trained on Case 3.  
This model used three hidden layers and the lbfgs solver.

### **5.4 Neural Network Decision Surface**
A 2D decision boundary plot was generated for Case 3 to show how the NN separated the input space.

---

## **6. Final Thoughts & Insights**
The project closes with a summary table comparing all models and a reflection on challenges, limitations of the feature sets, class imbalance, and the differences in how each model handled the problem.

---

## How to Run This Notebook
1. Clone the repository or download the notebook.  
2. Ensure you have Python 3 and the required libraries:
   - pandas  
   - numpy  
   - scikit-learn  
   - seaborn  
   - matplotlib  
3. Run the notebook in Jupyter, VS Code, or Google Colab.

---

## Files Included
- Jupyter Notebook (`.ipynb`) with all code and outputs  
- Decision Tree plot images for each case  
- README (this file)

---

## Notes
This project intentionally uses simple feature sets to highlight how model performance changes with different inputs. More complex feature engineering (e.g., using sex, passenger class, fare) would likely improve performance across all models.

