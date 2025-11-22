# Wine Quality Prediction: Ensemble Model Comparison Project

## Overview
This project compares several machine learning models on a dataset to explore which approach predicts wine quality most effectively. The goal is to understand model performance, identify strengths and weaknesses, and suggest improvements.

## Dataset
The dataset contains physicochemical properties of wines and their quality ratings.  
Source: [Wine Quality Dataset - UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/wine+quality)

## Models Tested
- Random Forest
- Decision Tree
- Support Vector Machine (SVM)
- Neural Network
- Voting Classifier (combination of DT, SVM, NN)

## Evaluation Metrics
- Accuracy
- F1 Score

## Results
| Model | Test Accuracy | Test F1 |
|-------|--------------|---------|
| Random Forest | 0.8875 | 0.866 |
| Decision Tree | 0.8125 | 0.819 |
| SVM | 0.85 | 0.822 |
| Neural Network | 0.8438 | 0.844 |
| Voting Classifier | 0.8625 | 0.849 |

## Project Goals
- Evaluate how different models perform on the dataset
- Understand why certain models perform better or worse
- Explore ways to improve predictive performance (e.g., feature engineering, hyperparameter tuning)

## Usage
1. Load the dataset.
2. Train and evaluate the models.
3. Review visualizations and metrics to compare model performance.

## Next Steps
- Tune model hyperparameters
- Explore additional features
- Consider advanced ensemble methods


