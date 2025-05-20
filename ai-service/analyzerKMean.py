import os
import json
import sys
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, StandardScaler
import coremltools as ct
from pathlib import Path
from sklearn.linear_model import LogisticRegression
import numpy as np  # Add this import at the top

# 1. Get the directory where THIS script lives
SCRIPT_DIR = Path(__file__).parent.resolve()

# 2. Construct absolute path to data
DATA_PATH = SCRIPT_DIR / "data" / "students_kmeans.json"  # Adjusted path

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
# def train_model(df):
#     # Encode categorical features
#     le = LabelEncoder() # check the purpose of it
#     X = df[['subject', 'avg_score']].copy()
#     X['subject'] = le.fit_transform(X['subject']) # check the purpose of it

#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(X)

#      # 3. Cluster students into 3 groups (adjustable)
#     kmeans = KMeans(n_clusters=3, random_state=42)
#     df['cluster'] = kmeans.fit_predict(X_scaled)


#     # 4. Identify which cluster represents under-performers
#     # (Assuming cluster with lowest avg_score is under-performers)
#     worst_cluster = df.groupby('cluster')['avg_score'].mean().idxmin()
#     df['critical'] = (df['cluster'] == worst_cluster).astype(int)

#     # 5. Convert to CoreML (using a surrogate model)
#     # KMeans isn't directly convertible, so we'll use a simple classifier
#     surrogate = LogisticRegression()
#     surrogate.fit(X_scaled, df['critical'])

#     coreml_model = ct.converters.sklearn.convert(
#         surrogate,
#         input_features=[
#             {'name': 'subject', 'type': 'int64'},
#             {'name': 'avg_score', 'type': 'double'}
#         ],
#         output_feature_names='needs_help'
#     )
#     coreml_model.save('StudentHelper.mlmodel')

#     return kmeans, le, scaler


def train_model(df, thresholds=None):
    """Enhanced KMeans training with configurable clustering"""
    try:
        # Set default thresholds if not provided
        if thresholds is None:
            thresholds = {
                "kmeans": {
                    "n_clusters": 3,
                    "features": ["avg_score", "trend"],
                    "subject_weight": 0.3,
                }
            }

        # 1. Feature Engineering
        le = LabelEncoder()
        
        # Prepare feature matrix
        X = df[["subject", "avg_score", "trend"]].copy()
        
        # Encode subjects
        X["subject_encoded"] = le.fit_transform(X["subject"])
        
        # Apply feature weights
        X["avg_score_weighted"] = X["avg_score"] * (1 - thresholds["kmeans"]["subject_weight"])
        X["subject_weighted"] = X["subject_encoded"] * thresholds["kmeans"]["subject_weight"]
        
        # Select only the weighted features for clustering
        X_features = X[["subject_weighted", "avg_score_weighted", "trend"]]
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_features)
        
        # 3. Fit KMeans with optimal clusters
        kmeans = KMeans(n_clusters=thresholds["kmeans"]["n_clusters"], random_state=42)
        df["cluster"] = kmeans.fit_predict(X_scaled)

        # 4. Cluster analysis
        cluster_profiles = (
            df.groupby("cluster")
            .agg({
                "avg_score": ["mean", "std"],
                "subject": lambda x: x.mode()[0],
                "student": "count",
            })
            .rename(columns={"student": "count"})
        )
        
				# Flatten the multi-index columns
        cluster_profiles.columns = ['_'.join(col).strip() for col in cluster_profiles.columns.values]
    
    			# Convert to dictionary
        cluster_dict = cluster_profiles.to_dict()

        # Identify worst cluster (lowest average score)
        # worst_cluster = cluster_profiles[("avg_score", "mean")].idxmin()
        worst_cluster = cluster_profiles["avg_score_mean"].idxmin()
        df["critical"] = (df["cluster"] == worst_cluster).astype(int)

        # Prepare insights
        insights = {
            "cluster_profiles": cluster_dict,
            "optimal_clusters": thresholds["kmeans"]["n_clusters"],
            "feature_weights": thresholds["kmeans"],
            "worst_cluster": int(worst_cluster)  # Explicitly convert to int
        }

        return kmeans, le, scaler, insights

    except Exception as e:
        raise Exception(f"KMeans training failed: {str(e)}")


# 3. Generate recommendations
def analyze(year):
    df, thresholds = load_data(year)

    try:
        kmeans, le, scaler, kmeans_insights = train_model(df, thresholds)
        worst_cluster = kmeans_insights["worst_cluster"]
        
        # Prepare prediction data with same encoding as training
        X_pred = df[["subject", "avg_score", "trend"]].copy()
        X_pred["subject_encoded"] = le.transform(X_pred["subject"])
        
        # Apply same feature weights as training
        X_pred["avg_score_weighted"] = X_pred["avg_score"] * (1 - thresholds["kmeans"]["subject_weight"])
        X_pred["subject_weighted"] = X_pred["subject_encoded"] * thresholds["kmeans"]["subject_weight"]
        
        # Scale features
        X_features = X_pred[["subject_weighted", "avg_score_weighted", "trend"]]
        X_scaled = scaler.transform(X_features)
        
        # Make predictions
        df["ml_critical"] = kmeans.predict(X_scaled) == worst_cluster
        
        # Calculate distances for confidence
        distances = kmeans.transform(X_scaled)
        with np.errstate(divide='ignore', invalid='ignore'):
            df["confidence"] = np.nan_to_num(1 - (distances.min(axis=1) / distances.max()), nan=0.0, posinf=1.0, neginf=0.0)

        ml_insights = {
            "method_used": "kmeans_clustering",
            "cluster_centers": kmeans.cluster_centers_.tolist(),
            "cluster_distribution": df["cluster"].value_counts().to_dict(),
            "confidence_metrics": {
                "min": float(df["confidence"].min()),
                "max": float(df["confidence"].max()),
                "mean": float(df["confidence"].mean()),
            },
            "kmeans_insights": kmeans_insights,
        }

    except Exception as e:
        ml_insights = {"error": str(e)}
        df["critical"] = 0  # Fallback column

    # Generate results
    critical_counts = df.groupby("student")["critical"].sum()
    
    return {
        "under_performing": critical_counts[
            critical_counts >= thresholds["min_critical_subjects"]
        ].index.tolist(),
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
