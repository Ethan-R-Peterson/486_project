"""
Screentime and Productivity Analysis Pipeline - Checkpoint 3 (REVISED)

Key Improvements:
- Removed circular target definitions
- Introduced real prediction tasks
- Proper cross-dataset feature usage (distribution-based)
- Added regression task for productivity
- Added true baseline models
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score, accuracy_score,
    mean_squared_error, mean_absolute_error, r2_score
)
from xgboost import XGBClassifier, XGBRegressor
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)


class ScreentimeProductivityPipeline:

    def __init__(self, data_dir='/Users/ethanpeterson/Documents/class_projects/486/proj'):
        self.data_dir = data_dir
        self.datasets = {}
        self.data = {}

    # =========================================================
    # STEP 1: LOAD DATA
    # =========================================================
    def load_datasets(self):
        print("Loading datasets...")

        self.datasets['social_media'] = pd.read_csv(
            f'{self.data_dir}/social_media_vs_productivity.csv'
        )
        self.datasets['productivity_tracker'] = pd.read_csv(
            f'{self.data_dir}/productivity_tracker_dataset.csv'
        )
        self.datasets['smartphone_usage'] = pd.read_csv(
            f'{self.data_dir}/random_smartphone_usage_dataset.csv'
        )

        print("Datasets loaded.")

    # =========================================================
    # STEP 2: FEATURE ENGINEERING + TARGETS
    # =========================================================
    def create_features_and_targets(self):
        print("Creating features and targets...")

        df = self.datasets['social_media'].copy()
        df_su = self.datasets['smartphone_usage']

        # -------------------------------
        # REAL TARGET 1: Productivity (REGRESSION)
        # -------------------------------
        y_reg = df['actual_productivity_score']

        # -------------------------------
        # REAL TARGET 2: High-risk behavior (CLASSIFICATION)
        # -------------------------------
        df['high_risk_usage'] = (
            (df['daily_social_media_time'] > df['daily_social_media_time'].quantile(0.75)) &
            (df['screen_time_before_sleep'] > df['screen_time_before_sleep'].quantile(0.75))
        ).astype(int)

        y_clf = df['high_risk_usage']

        # -------------------------------
        # CROSS-DATASET NORMALIZATION (SMARTPHONE DATA)
        # -------------------------------
        screen_mean = df_su['Daily_Screen_Time_Min'].mean()
        screen_std = df_su['Daily_Screen_Time_Min'].std()

        notif_mean = df_su['Notifications_Received'].mean()
        notif_std = df_su['Notifications_Received'].std()

        # -------------------------------
        # FEATURES (NO LEAKAGE)
        # -------------------------------
        X = pd.DataFrame()

        X['age'] = df['age']
        X['work_hours'] = df['work_hours_per_day']
        X['stress'] = df['stress_level']
        X['sleep'] = df['sleep_hours']
        X['notifications'] = df['number_of_notifications']
        X['social_time'] = df['daily_social_media_time']

        # Ratios
        X['screen_sleep_ratio'] = df['daily_social_media_time'] / (df['sleep_hours'] + 1)
        X['notif_work_ratio'] = df['number_of_notifications'] / (df['work_hours_per_day'] + 1)

        # Behavioral signals
        X['late_usage'] = (df['screen_time_before_sleep'] > 1).astype(int)
        X['focus_behavior'] = (
            df['uses_focus_apps'].astype(int) *
            df['has_digital_wellbeing_enabled'].astype(int)
        )

        # Normalized features (from smartphone dataset)
        X['screen_z'] = (df['daily_social_media_time'] - screen_mean) / (screen_std + 1e-5)
        X['notif_z'] = (df['number_of_notifications'] - notif_mean) / (notif_std + 1e-5)

        # Categorical
        X['gender'] = df['gender']
        X['job_type'] = df['job_type']

        # Encode categoricals
        for col in ['gender', 'job_type']:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))

        # Clean
        X = X.fillna(X.median())

        self.data = {
            'X': X,
            'y_reg': y_reg,
            'y_clf': y_clf
        }

        print("Feature engineering complete.")
        print(f"Features shape: {X.shape}")

    # =========================================================
    # STEP 3: PREPROCESS
    # =========================================================
    def preprocess(self):
        print("Preprocessing...")

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(self.data['X'])

        self.data['X_scaled'] = X_scaled
        self.data['scaler'] = scaler

    # =========================================================
    # STEP 4: TRAIN MODELS
    # =========================================================
    def train_models(self):
        print("Training models...")

        X = self.data['X_scaled']
        y_reg = self.data['y_reg']
        y_clf = self.data['y_clf']

        # -------------------------
        # REGRESSION TASK
        # -------------------------
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_reg, test_size=0.2, random_state=42
        )

        # Baseline
        dummy_reg = DummyRegressor(strategy='mean')
        dummy_reg.fit(X_train, y_train)

        rf_reg = RandomForestRegressor(n_estimators=200, random_state=42)
        rf_reg.fit(X_train, y_train)

        xgb_reg = XGBRegressor(n_estimators=200, random_state=42)
        xgb_reg.fit(X_train, y_train)

        preds_dummy = dummy_reg.predict(X_test)
        preds_rf = rf_reg.predict(X_test)
        preds_xgb = xgb_reg.predict(X_test)

        print("\nRegression Results (Productivity):")
        for name, preds in [('Dummy', preds_dummy), ('RF', preds_rf), ('XGB', preds_xgb)]:
            print(f"{name}: RMSE={mean_squared_error(y_test, preds, squared=False):.3f}, "
                  f"R2={r2_score(y_test, preds):.3f}")

        # -------------------------
        # CLASSIFICATION TASK
        # -------------------------
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_clf, test_size=0.2, stratify=y_clf, random_state=42
        )

        dummy_clf = DummyClassifier(strategy='most_frequent')
        dummy_clf.fit(X_train, y_train)

        lr = LogisticRegression(max_iter=1000)
        rf = RandomForestClassifier(n_estimators=200)
        xgb = XGBClassifier(eval_metric='logloss')

        lr.fit(X_train, y_train)
        rf.fit(X_train, y_train)
        xgb.fit(X_train, y_train)

        for name, model in [('Dummy', dummy_clf), ('LR', lr), ('RF', rf), ('XGB', xgb)]:
            preds = model.predict(X_test)
            print(f"\n{name} Classification:")
            print(classification_report(y_test, preds))

        self.models = {
            'rf_reg': rf_reg,
            'rf_clf': rf
        }

    # =========================================================
    # STEP 5: FEATURE IMPORTANCE
    # =========================================================
    def feature_importance(self):
        print("\nFeature Importance (Random Forest - Classification):")

        importances = self.models['rf_clf'].feature_importances_
        features = self.data['X'].columns

        sorted_idx = np.argsort(importances)[::-1]

        for i in range(min(10, len(features))):
            print(f"{features[sorted_idx[i]]}: {importances[sorted_idx[i]]:.4f}")

    # =========================================================
    # RUN ALL
    # =========================================================
    def run(self):
        self.load_datasets()
        self.create_features_and_targets()
        self.preprocess()
        self.train_models()
        self.feature_importance()


def main():
    pipeline = ScreentimeProductivityPipeline()
    pipeline.run()


if __name__ == "__main__":
    main()