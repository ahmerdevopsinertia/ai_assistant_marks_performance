import os
import json
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import coremltools as ct
from pathlib import Path

# 1. Get the directory where THIS script lives
SCRIPT_DIR = Path(__file__).parent.resolve()

# 2. Construct absolute path to data
DATA_PATH = SCRIPT_DIR / "data" / "generated_student_data_10.json"  # Adjusted path

# print(f"Debug Info:")
# print(f"Python Executable: {sys.executable}")
# print(f"VENV Active: {'venv' in sys.executable}")
# print(f"Current Working Dir: {os.getcwd()}")
# print(f"File Location: {__file__}")


# 1. Load and prepare data
def load_data(year):
    with open(DATA_PATH) as f:
        raw_data = json.load(f)[f"academic_year_{year}"]

    # Convert to Pandas DataFrame
    records = []
    for student, details in raw_data["students"].items():
        for subject, grades in details["subjects"].items():
            records.append(
                {
                    "student": student,
                    "subject": subject,
                    "avg_score": sum(grades) / len(grades),
                    "trend": 1 if grades[-1] > grades[0] else 0,  # 1=improving
                }
            )

    df = pd.DataFrame(records)
    return df, raw_data["thresholds"]


# 2. Train prediction model
def train_model(df):
    # Encode categorical features
    le = LabelEncoder()  # check the purpose of it
    X = df[["subject", "avg_score"]].copy()
    X["subject"] = le.fit_transform(X["subject"])  # check the purpose of it

    # Predict if student needs intervention (1=needs help)
    y = (df["avg_score"] < df["avg_score"].quantile(0.3)).astype(
        int
    )  # check the purpose of it

    model = RandomForestClassifier(n_estimators=50, max_depth=3)
    model.fit(X, y)  # check the purpose of it

    # Convert to Core ML for iOS compatibility
    coreml_model = ct.converters.sklearn.convert(
        model,
        input_features=["subject", "avg_score"],
        output_feature_names="needs_help",
    )  # check the purpose of it
    coreml_model.save("StudentHelper.mlmodel")  # check the purpose of it

    return model, le


# 3. Generate recommendations
def analyze(year):
    df, thresholds = load_data(year)

    # 1. Set defaults
    thresholds.setdefault("min_critical_subjects", 1)
    thresholds.setdefault("use_ml_suggestions", True)
    thresholds.setdefault("ml_score_range", [20, 80])
    thresholds.setdefault("under_performing", 60)  # Add default if missing

    # 2. Calculate overall averages per student
    student_overall_avg = df.groupby("student")["avg_score"].mean()

    # 3. Apply absolute thresholds
    df["critical"] = df.apply(
        lambda row: (
            1
            if row["avg_score"] < thresholds["subject_thresholds"][row["subject"]]
            else 0
        ),
        axis=1,
    )

    # 4. Conditional ML analysis
    ml_insights = {"method_used": "absolute_thresholds"}
    if (
        thresholds["use_ml_suggestions"]
        and df["avg_score"].between(*thresholds["ml_score_range"]).any()
    ):
        try:
            model, le = train_model(df)
            X_pred = df[["subject", "avg_score"]].copy()
            X_pred["subject"] = le.transform(X_pred["subject"])
            df["ml_critical"] = model.predict(X_pred)

            # Get prediction probabilities
            try:
                probas = model.predict_proba(X_pred)
                confidence = (
                    probas[:, 1] if probas.shape[1] == 2 else probas.max(axis=1)
                )
            except AttributeError:
                confidence = np.ones(len(X_pred)) * 0.5

            ml_insights.update(
                {
                    "feature_importances": dict(
                        zip(["subject", "avg_score"], model.feature_importances_)
                    ),
                    "method_used": "hybrid_threshold_ml",
                    "confidence_metrics": {
                        "min": float(np.min(confidence)),
                        "max": float(np.max(confidence)),
                        "mean": float(np.mean(confidence)),
                    },
                    "tree_votes": {
                        "total_trees": getattr(model, "n_estimators", 1),
                        "avg_agree": float(
                            np.mean(confidence) * getattr(model, "n_estimators", 1)
                        ),
                    },
                }
            )

            # Combine ML predictions with thresholds
            df["critical"] = df.apply(
                lambda row: (
                    1 if (row["critical"] == 1) or (row["ml_critical"] == 1) else 0
                ),
                axis=1,
            )

        except Exception as e:
            ml_insights["ml_error"] = str(e)

    # 5. Generate results with proper under_performing check
    critical_counts = df.groupby("student")["critical"].sum()
    critical_students = critical_counts[
        critical_counts >= thresholds["min_critical_subjects"]
    ].index.tolist()

    # Apply under_performing threshold correctly
    under_performing = [
        student
        for student in critical_students
        if student_overall_avg[student] < thresholds["under_performing"]
    ]

    return {
        "under_performing": under_performing,
        "subject_analysis": {
            subject: {
                "critical_students": df[
                    (df["subject"] == subject) & (df["critical"] == 1)
                ][["student", "avg_score"]].to_dict("records"),
                "class_avg": df[df["subject"] == subject]["avg_score"].mean(),
            }
            for subject in thresholds["subject_thresholds"]
        },
        "analysis_metadata": ml_insights,
    }


def generate_actions(subject, trend):
    base_actions = {
        "math": ["Daily practice: 5 algebra problems", "Use visual geometry tools"],
        "science": ["Hands-on experiments weekly", "Concept mapping before tests"],
    }
    if not trend:
        base_actions[subject].append("Request teacher consultation")
    return base_actions.get(subject, ["General tutoring recommended"])


if __name__ == "__main__":
    year = sys.argv[1]
    try:
        result = analyze(year)
        # Only print the final JSON output
        print(json.dumps(result))
    except Exception as e:
        # Error handling as JSON
        print(json.dumps({"error": str(e)}))
