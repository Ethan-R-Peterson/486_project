"""
Social Media & Mental Health Pipeline - Revised (Checkpoint 3+)

Predicts Addiction Level and Productivity Loss from the Time-Wasters on
Social Media dataset, with cross-dataset feature enrichment from the
smartphone-usage and productivity-tracker datasets.

Key design choices:
  - Primary dataset: Time-Wasters on Social Media (1 000 rows, 31 columns)
  - Targets: Addiction Level (0-7, classification) and ProductivityLoss (1-9, regression/classification)
  - Leaked columns removed: Self Control, Satisfaction (deterministic transforms of the targets)
  - Cross-dataset enrichment: population-level statistics from the smartphone
    and productivity-tracker datasets are joined as normalizing references
  - Baselines: DummyClassifier / DummyRegressor so we can prove the models
    actually learn signal
  - Evaluation: Accuracy, macro-F1, per-class classification reports, and
    cross-validation
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OrdinalEncoder
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_validate
)
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    RandomForestRegressor
)
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score,
    accuracy_score, mean_squared_error, r2_score
)
import warnings

warnings.filterwarnings("ignore")
np.random.seed(42)


# ─────────────────────────────────────────────
# HELPER: Print a section header
# ─────────────────────────────────────────────
def header(text: str) -> str:
    line = "=" * 64
    block = f"\n{line}\n  {text}\n{line}"
    print(block)
    return block


class SocialMediaMentalHealthPipeline:
    """End-to-end ML pipeline for predicting social-media addiction and
    productivity loss."""

    def __init__(self, data_dir: str = "."):
        self.data_dir = data_dir
        self.datasets: dict[str, pd.DataFrame] = {}
        self.results_log: list[str] = []  # collect all printed output

    def _log(self, text: str):
        """Print *and* store for the results file."""
        print(text)
        self.results_log.append(text)

    # =========================================================
    # STEP 1 — LOAD DATA
    # =========================================================
    def load_datasets(self):
        header_text = header("STEP 1: Loading Datasets")
        self.results_log.append(header_text)

        self.datasets["tw"] = pd.read_csv(
            f"{self.data_dir}/Time_Wasters_on_Social_Media.csv"
        )
        self.datasets["sm"] = pd.read_csv(
            f"{self.data_dir}/social_media_vs_productivity.csv"
        )
        self.datasets["pt"] = pd.read_csv(
            f"{self.data_dir}/productivity_tracker_dataset.csv"
        )
        self.datasets["su"] = pd.read_csv(
            f"{self.data_dir}/random_smartphone_usage_dataset.csv"
        )

        for name, df in self.datasets.items():
            self._log(f"  {name:20s} → {df.shape[0]:>6,} rows × {df.shape[1]:>2} cols")

    # =========================================================
    # STEP 2 — EXPLORATORY DATA ANALYSIS
    # =========================================================
    def eda(self):
        header_text = header("STEP 2: Exploratory Data Analysis")
        self.results_log.append(header_text)

        tw = self.datasets["tw"]

        self._log("\nTarget: Addiction Level distribution")
        for val, cnt in tw["Addiction Level"].value_counts().sort_index().items():
            self._log(f"  Level {val}: {cnt:>4}  ({cnt / len(tw) * 100:5.1f}%)")

        self._log("\nTarget: ProductivityLoss distribution")
        for val, cnt in tw["ProductivityLoss"].value_counts().sort_index().items():
            self._log(f"  Loss {val}: {cnt:>4}  ({cnt / len(tw) * 100:5.1f}%)")

        # Note deterministic leakage
        self._log("\n⚠  Leakage check:")
        self._log("   Addiction Level == 10 − Self Control   (exact)")
        self._log("   ProductivityLoss == 10 − Satisfaction  (exact)")
        self._log("   → Both columns EXCLUDED from features.\n")

    # =========================================================
    # STEP 3 — FEATURE ENGINEERING
    # =========================================================
    def create_features_and_targets(self):
        header_text = header("STEP 3: Feature Engineering")
        self.results_log.append(header_text)

        tw = self.datasets["tw"].copy()
        su = self.datasets["su"]
        pt = self.datasets["pt"]
        sm = self.datasets["sm"]

        # ── Targets ──────────────────────────────────────────
        y_addiction = tw["Addiction Level"]           # 0-7 classification
        y_productivity = tw["ProductivityLoss"] - 1    # shift to 0-8 for XGBoost compatibility

        # ── Drop leaked / ID columns ────────────────────────
        drop_cols = [
            "UserID", "Video ID",                    # IDs
            "Self Control", "Satisfaction",           # deterministic leakage
            "Addiction Level", "ProductivityLoss",    # targets
        ]
        tw.drop(columns=drop_cols, inplace=True)

        # ── Parse Watch Time → hour of day ──────────────────
        def parse_hour(t: str) -> int:
            """'9:00 PM' → 21, '2:00 AM' → 2"""
            try:
                parts = t.strip().split()
                h = int(parts[0].split(":")[0])
                if parts[1].upper() == "PM" and h != 12:
                    h += 12
                elif parts[1].upper() == "AM" and h == 12:
                    h = 0
                return h
            except Exception:
                return -1

        tw["watch_hour"] = tw["Watch Time"].apply(parse_hour)
        tw["is_late_night"] = tw["watch_hour"].apply(
            lambda h: 1 if h >= 21 or h <= 5 else 0
        )
        tw.drop(columns=["Watch Time"], inplace=True)

        # ── Engineered behavioural features ─────────────────
        tw["time_per_session"] = tw["Total Time Spent"] / (tw["Number of Sessions"] + 1)
        tw["time_per_video"] = tw["Total Time Spent"] / (tw["Number of Videos Watched"] + 1)
        tw["engagement_rate"] = tw["Engagement"] / (tw["Total Time Spent"] + 1)
        tw["scroll_per_session"] = tw["Scroll Rate"] / (tw["Number of Sessions"] + 1)
        tw["video_completion"] = tw["Time Spent On Video"] / (tw["Video Length"] + 1)

        # ── Cross-dataset enrichment ────────────────────────
        # 1) Smartphone dataset: population-level screen-time stats by app category
        app_stats = (
            su.groupby("Primary_App_Category")["Daily_Screen_Time_Min"]
            .agg(["mean", "std"])
            .rename(columns={"mean": "pop_screen_mean", "std": "pop_screen_std"})
        )
        # Map TW "Video Category" → broad app category (best-effort)
        cat_map = {
            "Gaming": "Gaming", "Pranks": "Social", "Vlogs": "Social",
            "Tutorials": "Productivity", "News": "Productivity",
            "Entertainment": "Entertainment", "Cooking": "Productivity",
            "Music": "Entertainment", "Sports": "Entertainment",
        }
        tw["app_cat_mapped"] = tw["Video Category"].map(cat_map).fillna("Social")
        tw = tw.merge(app_stats, left_on="app_cat_mapped",
                       right_index=True, how="left")
        tw["screen_z_pop"] = (
            (tw["Total Time Spent"] - tw["pop_screen_mean"])
            / (tw["pop_screen_std"] + 1e-5)
        )
        tw.drop(columns=["app_cat_mapped", "pop_screen_mean", "pop_screen_std"],
                inplace=True)

        # 2) Productivity-tracker dataset: avg productivity & screen time by occupation
        #    Map TW professions to PT occupations (partial overlap)
        occ_prod = (
            pt.groupby("Occupation")
            .agg(occ_avg_productivity=("ProductivityScore", "mean"),
                 occ_avg_screentime=("ScreenTimeHours", "mean"))
        )
        prof_to_occ = {
            "Engineer": "Engineer", "Students": "Student",
            "Teacher": "Teacher", "Artist": "Artist",
        }
        tw["occ_mapped"] = tw["Profession"].map(prof_to_occ)
        tw = tw.merge(occ_prod, left_on="occ_mapped",
                       right_index=True, how="left")
        # Fill unmatched professions with global averages
        tw["occ_avg_productivity"].fillna(pt["ProductivityScore"].mean(), inplace=True)
        tw["occ_avg_screentime"].fillna(pt["ScreenTimeHours"].mean(), inplace=True)
        tw.drop(columns=["occ_mapped"], inplace=True)

        # 3) Social-media-vs-productivity dataset: population-level stress & sleep stats
        #    No direct profession match, so use global averages as reference benchmarks
        pop_stress_mean = sm["stress_level"].mean()
        pop_sleep_mean = sm["sleep_hours"].mean()
        pop_social_time_mean = sm["daily_social_media_time"].mean()
        pop_social_time_std = sm["daily_social_media_time"].std()

        # Z-score the TW user's total time against the SM population
        tw["social_time_z_sm"] = (
            (tw["Total Time Spent"] - pop_social_time_mean)
            / (pop_social_time_std + 1e-5)
        )
        # Add population reference values as constant features (useful for
        # interaction with per-user features in tree models)
        tw["pop_stress_ref"] = pop_stress_mean
        tw["pop_sleep_ref"] = pop_sleep_mean

        # ── Encode categoricals ─────────────────────────────
        cat_cols = tw.select_dtypes(include=["object", "bool"]).columns.tolist()
        self._log(f"  Categorical columns to encode: {cat_cols}")

        label_encoders = {}
        for col in cat_cols:
            le = LabelEncoder()
            tw[col] = le.fit_transform(tw[col].astype(str))
            label_encoders[col] = le

        # ── Final clean ─────────────────────────────────────
        tw = tw.fillna(tw.median(numeric_only=True))

        self._log(f"  Final feature matrix: {tw.shape}")
        self._log(f"  Features: {list(tw.columns)}")

        self.X = tw
        self.y_addiction = y_addiction
        self.y_productivity = y_productivity
        self.label_encoders = label_encoders

    # =========================================================
    # STEP 4 — PREPROCESSING (SCALE)
    # =========================================================
    def preprocess(self):
        header_text = header("STEP 4: Preprocessing")
        self.results_log.append(header_text)

        self.scaler = StandardScaler()
        self.X_scaled = self.scaler.fit_transform(self.X)
        self._log(f"  Scaled feature matrix: {self.X_scaled.shape}")

    # =========================================================
    # STEP 5 — TRAIN & EVALUATE
    # =========================================================
    def train_and_evaluate(self):
        # ─── TASK A: Addiction Level (multi-class classification) ───
        header_text = header("STEP 5A: Addiction Level — Classification")
        self.results_log.append(header_text)

        X = self.X_scaled
        y = self.y_addiction

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )

        models = {
            "Baseline (most-frequent)": DummyClassifier(strategy="most_frequent"),
            "Baseline (stratified)":    DummyClassifier(strategy="stratified"),
            "Logistic Regression":      LogisticRegression(max_iter=2000),
            "Random Forest":            RandomForestClassifier(n_estimators=300, random_state=42),
            "Gradient Boosting":        GradientBoostingClassifier(
                                            n_estimators=300, max_depth=6,
                                            learning_rate=0.1,
                                            random_state=42
                                        ),
        }

        self._log(f"\n  Train size: {len(y_train)}  |  Test size: {len(y_test)}")

        best_f1, best_name = -1, ""
        for name, model in models.items():
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            acc = accuracy_score(y_test, preds)
            f1 = f1_score(y_test, preds, average="macro")
            self._log(f"\n  ── {name} ──")
            self._log(f"     Accuracy : {acc:.4f}")
            self._log(f"     Macro-F1 : {f1:.4f}")
            if "Baseline" not in name:
                self._log(f"\n{classification_report(y_test, preds)}")
            if f1 > best_f1:
                best_f1, best_name = f1, name

        self._log(f"\n  ★ Best model (Addiction): {best_name}  (Macro-F1 = {best_f1:.4f})")

        # Cross-validation on best non-baseline
        self._log("\n  5-fold stratified CV on best model:")
        best_model = models[best_name]
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_results = cross_validate(
            best_model, X, y, cv=cv,
            scoring=["accuracy", "f1_macro"],
            return_train_score=False
        )
        self._log(f"     CV Accuracy : {cv_results['test_accuracy'].mean():.4f} ± {cv_results['test_accuracy'].std():.4f}")
        self._log(f"     CV Macro-F1 : {cv_results['test_f1_macro'].mean():.4f} ± {cv_results['test_f1_macro'].std():.4f}")

        self.best_clf_addiction = best_model

        # ─── TASK B: Productivity Loss (multi-class classification) ───
        header_text = header("STEP 5B: Productivity Loss — Classification")
        self.results_log.append(header_text)

        y2 = self.y_productivity

        X_train2, X_test2, y_train2, y_test2 = train_test_split(
            X, y2, test_size=0.2, stratify=y2, random_state=42
        )

        models2 = {
            "Baseline (most-frequent)": DummyClassifier(strategy="most_frequent"),
            "Baseline (stratified)":    DummyClassifier(strategy="stratified"),
            "Logistic Regression":      LogisticRegression(max_iter=2000),
            "Random Forest":            RandomForestClassifier(n_estimators=300, random_state=42),
            "Gradient Boosting":        GradientBoostingClassifier(
                                            n_estimators=300, max_depth=6,
                                            learning_rate=0.1,
                                            random_state=42
                                        ),
        }

        self._log(f"\n  Train size: {len(y_train2)}  |  Test size: {len(y_test2)}")

        best_f1_2, best_name_2 = -1, ""
        for name, model in models2.items():
            model.fit(X_train2, y_train2)
            preds = model.predict(X_test2)
            acc = accuracy_score(y_test2, preds)
            f1 = f1_score(y_test2, preds, average="macro")
            self._log(f"\n  ── {name} ──")
            self._log(f"     Accuracy : {acc:.4f}")
            self._log(f"     Macro-F1 : {f1:.4f}")
            if "Baseline" not in name:
                self._log(f"\n{classification_report(y_test2, preds)}")
            if f1 > best_f1_2:
                best_f1_2, best_name_2 = f1, name

        self._log(f"\n  ★ Best model (Productivity Loss): {best_name_2}  (Macro-F1 = {best_f1_2:.4f})")

        # Cross-validation
        self._log("\n  5-fold stratified CV on best model:")
        best_model2 = models2[best_name_2]
        cv_results2 = cross_validate(
            best_model2, X, y2, cv=cv,
            scoring=["accuracy", "f1_macro"],
            return_train_score=False
        )
        self._log(f"     CV Accuracy : {cv_results2['test_accuracy'].mean():.4f} ± {cv_results2['test_accuracy'].std():.4f}")
        self._log(f"     CV Macro-F1 : {cv_results2['test_f1_macro'].mean():.4f} ± {cv_results2['test_f1_macro'].std():.4f}")

        self.best_clf_productivity = best_model2

    # =========================================================
    # STEP 6 — FEATURE IMPORTANCE
    # =========================================================
    def feature_importance(self):
        header_text = header("STEP 6: Feature Importance")
        self.results_log.append(header_text)

        for task_name, model in [
            ("Addiction Level", self.best_clf_addiction),
            ("Productivity Loss", self.best_clf_productivity),
        ]:
            if not hasattr(model, "feature_importances_"):
                continue
            self._log(f"\n  Top 10 features — {task_name}:")
            importances = model.feature_importances_
            features = self.X.columns
            sorted_idx = np.argsort(importances)[::-1]
            for i in range(min(10, len(features))):
                idx = sorted_idx[i]
                self._log(f"    {i+1:>2}. {features[idx]:30s} {importances[idx]:.4f}")

    # =========================================================
    # STEP 7 — SAVE RESULTS
    # =========================================================
    def save_results(self):
        header_text = header("STEP 7: Saving Results")
        self.results_log.append(header_text)

        path = f"{self.data_dir}/pipe3_results.txt"
        with open(path, "w") as f:
            f.write("\n".join(self.results_log))
        self._log(f"  Results saved to {path}")

    # =========================================================
    # RUN ALL
    # =========================================================
    def run(self):
        self.load_datasets()
        self.eda()
        self.create_features_and_targets()
        self.preprocess()
        self.train_and_evaluate()
        self.feature_importance()
        self.save_results()


def main():
    pipeline = SocialMediaMentalHealthPipeline(data_dir="/Users/taleefe/Downloads/486 Project")
    pipeline.run()


if __name__ == "__main__":
    main()