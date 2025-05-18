import json
import sys
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import coremltools as ct
from pathlib import Path

# 1. Get the directory where THIS script lives
SCRIPT_DIR = Path(__file__).parent.resolve()

# 2. Construct absolute path to data
DATA_PATH = SCRIPT_DIR / 'data' / 'students.json'  # Adjusted path



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
def train_model(df):
    # Encode categorical features
    le = LabelEncoder()
    X = df[['subject', 'avg_score']].copy()
    X['subject'] = le.fit_transform(X['subject'])
    
    # Predict if student needs intervention (1=needs help)
    y = (df['avg_score'] < df['avg_score'].quantile(0.3)).astype(int)
    
    model = RandomForestClassifier(n_estimators=50, max_depth=3)
    model.fit(X, y)
    
    # Convert to Core ML for iOS compatibility
    coreml_model = ct.converters.sklearn.convert(
        model,
        input_features=['subject', 'avg_score'],
        output_feature_names='needs_help'
    )
    coreml_model.save('StudentHelper.mlmodel')
    
    return model, le

# 3. Generate recommendations
def analyze(year):
    df, thresholds = load_data(year)
    model, le = train_model(df)
    
    results = {
        'under_performing': [],
        'subject_analysis': {},
        'model_metrics': {
            'feature_importances': dict(zip(
                ['subject', 'avg_score'],
                model.feature_importances_
            ))
        }
    }
    
    # Identify critical cases
    X_pred = df[['subject', 'avg_score']].copy()
    X_pred['subject'] = le.transform(X_pred['subject'])
    df['prediction'] = model.predict(X_pred)
    
    # Group by subject
    for subject in df['subject'].unique():
        subject_df = df[df['subject'] == subject]
        critical_cases = subject_df[subject_df['prediction'] == 1]
        
        results['subject_analysis'][subject] = {
            'critical_students': [
                {
                    'name': row['student'],
                    'score': row['avg_score'],
                    'action_items': generate_actions(subject, row['trend'])
                }
                for _, row in critical_cases.iterrows()
            ],
            'class_avg': subject_df['avg_score'].mean()
        }
    
    # Overall under-performers (multiple weak subjects)
    student_scores = df.groupby('student')['prediction'].sum()
    results['under_performing'] = student_scores[student_scores >= 2].index.tolist()
    
    return results

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