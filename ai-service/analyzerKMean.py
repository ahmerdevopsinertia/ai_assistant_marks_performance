import os
import json
import sys
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, StandardScaler
import coremltools as ct
from pathlib import Path
from sklearn.linear_model import LogisticRegression

# 1. Get the directory where THIS script lives
SCRIPT_DIR = Path(__file__).parent.resolve()

# 2. Construct absolute path to data
DATA_PATH = SCRIPT_DIR / 'data' / 'students_kmeans.json'  # Adjusted path

print(f"Debug Info:")
print(f"Python Executable: {sys.executable}")
print(f"VENV Active: {'venv' in sys.executable}")
print(f"Current Working Dir: {os.getcwd()}")
print(f"File Location: {__file__}")


# 1. Load and prepare data
def load_data(year):
    with open(DATA_PATH) as f:
        raw_data = json.load(f)[f'academic_year_{year}']
    
    # Convert to Pandas DataFrame
    records = []
    for student, details in raw_data['students'].items():
        for subject, grades in details['subjects'].items():
            records.append({
                'student': student,
                'subject': subject,
                'avg_score': sum(grades)/len(grades),
                'trend': 1 if grades[-1] > grades[0] else 0  # 1=improving
            })
    
    df = pd.DataFrame(records)
    return df, raw_data['thresholds']

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
    # Set default thresholds if not provided
    if thresholds is None:
        thresholds = {
            'kmeans': {
                'n_clusters': 3,
                'features': ['avg_score', 'volatility'],
                'subject_weight': 0.3
            }
        }
    
    # 1. Feature Engineering
    le = LabelEncoder()
    
    # Calculate additional features
    df['volatility'] = df.apply(lambda row: max(row['grades'])-min(row['grades']), axis=1)
    df['trend'] = df['grades'].apply(lambda x: 1 if x[-1] > x[0] else 0)
    
    # Prepare feature matrix with configurable weights
    X = df[['subject', 'avg_score', 'volatility']].copy()
    X['subject'] = le.fit_transform(X['subject'])
    X['avg_score'] = X['avg_score'] * (1 - thresholds['kmeans']['subject_weight'])
    X['subject'] = X['subject'] * thresholds['kmeans']['subject_weight']
    
    # 2. Determine optimal cluster count
    def find_optimal_clusters(X, max_k=5):
        wcss = []
        for k in range(1, max_k+1):
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(X)
            wcss.append(kmeans.inertia_)
        
        # Simple elbow detection (could be enhanced)
        optimal_k = thresholds['kmeans'].get('n_clusters', 3)
        if len(wcss) > 3:
            deltas = [wcss[i]-wcss[i+1] for i in range(len(wcss)-1)]
            optimal_k = deltas.index(max(deltas)) + 1
        return optimal_k
    
    n_clusters = find_optimal_clusters(X)
    
    # 3. Fit KMeans with optimal clusters
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['cluster'] = kmeans.fit_predict(X_scaled)
    
    # 4. Cluster analysis
    cluster_profiles = df.groupby('cluster').agg({
        'avg_score': ['mean', 'std'],
        'volatility': 'mean',
        'subject': lambda x: le.inverse_transform(x.mode())[0],
        'student': 'count'
    }).rename(columns={'student': 'count'})
    
    # Identify worst cluster (configurable method)
    worst_cluster = cluster_profiles[('avg_score', 'mean')].idxmin()
    df['critical'] = (df['cluster'] == worst_cluster).astype(int)
    
    # 5. CoreML conversion with surrogate model
    surrogate = LogisticRegression()
    surrogate.fit(X_scaled, df['critical'])
    
    coreml_model = ct.converters.sklearn.convert(
        surrogate,
        input_features=[
            {'name': 'subject', 'type': 'int64'},
            {'name': 'avg_score', 'type': 'double'},
            {'name': 'volatility', 'type': 'double'}
        ],
        output_feature_names='needs_help'
    )
    coreml_model.save('StudentHelper.mlmodel')
    
    # Prepare insights
    insights = {
        'cluster_profiles': cluster_profiles.to_dict(),
        'optimal_clusters': n_clusters,
        'feature_weights': {
            'subject': thresholds['kmeans']['subject_weight'],
            'score_volatility': 1 - thresholds['kmeans']['subject_weight']
        }
    }
    
    return kmeans, le, scaler, insights


# 3. Generate recommendations
def analyze(year):
  
    df, thresholds = load_data(year)
    
        # KMeans analysis
    try:
        kmeans, le, scaler, kmeans_insights = train_model(df)
        X = df[['subject', 'avg_score']].copy()
        X['subject'] = le.transform(X['subject'])
        X_scaled = scaler.transform(X)
        
        df['ml_critical'] = kmeans.predict(X_scaled) == kmeans.cluster_centers_[:,1].argmin()
        
        # Distance to cluster center as confidence metric
        distances = kmeans.transform(X_scaled)
        df['confidence'] = 1 - (distances.min(axis=1) / distances.max())
        
        ml_insights = {
            'method_used': 'kmeans_clustering',
            'cluster_centers': kmeans.cluster_centers_.tolist(),
            'cluster_distribution': df['cluster'].value_counts().to_dict(),
            'confidence_metrics': {
                'min': float(df['confidence'].min()),
                'max': float(df['confidence'].max()),
                'mean': float(df['confidence'].mean())
            },
            'kmeans_insights': kmeans_insights
        }
        
    except Exception as e:
        ml_insights = {'error': str(e)}
    
    # 4. Generate results
    critical_counts = df.groupby('student')['critical'].sum()
    return {
        'under_performing': critical_counts[
            critical_counts >= thresholds['min_critical_subjects']
        ].index.tolist(),
        'subject_analysis': {
            subject: {
                'critical_students': df[
                    (df['subject'] == subject) & 
                    (df['critical'] == 1)
                ][['student', 'avg_score']].to_dict('records'),
                'class_avg': df[df['subject'] == subject]['avg_score'].mean()
            }
            for subject in thresholds['subject_thresholds']
        },
        'analysis_metadata': ml_insights
    }

def generate_actions(subject, trend):
    base_actions = {
        'math': [
            "Daily practice: 5 algebra problems",
            "Use visual geometry tools"
        ],
        'science': [
            "Hands-on experiments weekly",
            "Concept mapping before tests"
        ]
    }
    if not trend:
        base_actions[subject].append("Request teacher consultation")
    return base_actions.get(subject, ["General tutoring recommended"])

if __name__ == '__main__':
    year = sys.argv[1]
    try:
        result = analyze(year)
        # Only print the final JSON output
        print(json.dumps(result))
    except Exception as e:
        # Error handling as JSON
        print(json.dumps({"error": str(e)}))