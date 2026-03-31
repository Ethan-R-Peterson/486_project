# Pipe2.py Explanation

## Overview
`pipe2.py` implements a machine learning pipeline for analyzing screentime and productivity data. It processes three datasets to predict productivity scores (regression) and identify high-risk usage behavior (classification).

## Datasets Used
- **social_media_vs_productivity.csv**: Main dataset with user demographics, screentime habits, and productivity scores.
- **productivity_tracker_dataset.csv**: Additional productivity tracking data (loaded but not directly used in features).
- **random_smartphone_usage_dataset.csv**: Used for cross-dataset normalization of screentime and notification features.

## Pipeline Steps

### 1. Load Datasets
Loads the three CSV files into pandas DataFrames.

### 2. Feature Engineering and Targets
- **Regression Target**: `actual_productivity_score` from the social media dataset.
- **Classification Target**: `high_risk_usage` - binary label for users with high screentime (>75th percentile) and late screen usage (>75th percentile).
- **Features**: Includes age, work hours, stress, sleep, notifications, social media time, ratios (e.g., screen/sleep), behavioral signals, normalized features from smartphone data, and encoded categoricals (gender, job type).
- Handles missing values by filling with medians and removes rows with NaN targets.

### 3. Preprocessing
Applies StandardScaler to normalize features.

### 4. Train Models
- **Regression**: Predicts productivity score using Dummy (mean), Random Forest, and XGBoost regressors. Evaluates with RMSE and R².
- **Classification**: Predicts high-risk usage using Dummy (most frequent), Logistic Regression, Random Forest, and XGBoost classifiers. Evaluates with classification report.
- Uses 80/20 train-test split with stratification for classification.

### 5. Feature Importance
Displays top 10 features by importance from the Random Forest classifier.

### 6. Save Results
Outputs a summary to `pipe2_results.txt` including dataset shapes, feature importance, and notes on metrics.

## Key Improvements
- Avoids data leakage by using cross-dataset normalization.
- Includes baseline models for comparison.
- Handles imbalanced classification implicitly through stratification.

## Dependencies
- pandas, numpy, scikit-learn, xgboost

Run with: `python pipe2.py`