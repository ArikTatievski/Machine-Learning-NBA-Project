# NBA Player Ratings Prediction

## Overview

This project aims to answer the question: Can we predict the rating of new incoming NBA players or the rating of all NBA players in the next year based on their salaries? We explore this question using machine learning models, specifically Support Vector Machines (SVM), Random Forest, AdaBoost, and k-Nearest Neighbors (KNN).

## Dataset

The dataset used for this project is sourced from [Kaggle](https://www.kaggle.com/datasets/omarsobhy14/nba-players-salaries?resource=download), with an additional column added for 2K-Rating. The properties of the dataset include:

- **Player Id**: Integer
- **Player Name**: String
- **2022/2023 Salary**: Integer + '$'
- **2023/2024 Salary**: Integer + '$'
- **2024/2025 Salary**: Integer + '$'
- **2K-Rating**: Integer (Range: 60-99)

## Data Split

The dataset was split into 50% training and 50% testing to evaluate the model performance effectively.

## Models Used

1. **Support Vector Machines (SVM) Model**
2. **Random Forest Model**
3. **AdaBoost Model**
4. **k-Nearest Neighbors (KNN) Model**

## Approach

KNN yielded the best results, which we attribute to its local proximity nature and effective parameter tuning. The project utilizes this model to predict NBA player ratings based on their salaries.
