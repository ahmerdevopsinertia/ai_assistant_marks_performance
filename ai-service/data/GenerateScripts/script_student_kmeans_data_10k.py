import json
import random
from faker import Faker
import numpy as np

# Initialize faker for random names
fake = Faker()

def generate_student_data(num_students=10000):
    # Base structure
    data = {
        "academic_year_2024": {
            "thresholds": {
                "min_critical_subjects": 1,
                "subject_thresholds": ["math", "science"],
                "kmeans": {
                    "n_clusters": 3,
                    "features": ["avg_score", "trend"],
                    "subject_weight": 0.3
                }
            },
            "students": {}
        }
    }

    # Performance categories with realistic distributions
    performance_groups = [
        # (group_size_percentage, math_range, science_range)
        (0.01, (85, 100), (85, 100)),   # Top performers (1%)
        (0.04, (70, 89), (70, 89)),      # Good performers (4%)
        (0.15, (55, 75), (55, 75)),      # Average (15%)
        (0.10, (40, 60), (40, 60)),      # Below average (10%)
        (0.10, (30, 50), (30, 50)),      # Struggling (10%)
        (0.60, (1, 40), (1, 40))         # Under-performers (60%)
    ]

    # Generate student records
    for i in range(num_students):
        # Determine performance group
        group = None
        rand = random.random()
        cumulative = 0
        for percent, math_r, sci_r in performance_groups:
            cumulative += percent
            if rand <= cumulative:
                group = (math_r, sci_r)
                break

        # Generate scores with realistic trends
        math_scores = [
            random.randint(group[0][0], group[0][1]),
            random.randint(group[0][0], group[0][1]),
            random.randint(group[0][0], group[0][1])
        ]
        
        science_scores = [
            random.randint(group[1][0], group[1][1]),
            random.randint(group[1][0], group[1][1]),
            random.randint(group[1][0], group[1][1])
        ]

        # Add realistic trends
        trend = random.choice(['improving', 'declining', 'stable'])
        if trend == 'improving':
            math_scores.sort()
            science_scores.sort()
        elif trend == 'declining':
            math_scores.sort(reverse=True)
            science_scores.sort(reverse=True)

        # Generate unique username
        username = fake.unique.user_name()
        
        # Add student record
        data["academic_year_2024"]["students"][username] = {
            "subjects": {
                "math": math_scores,
                "science": science_scores
            }
        }

    return data

# Generate the data
student_data = generate_student_data()

# Save to file
with open('generated_student_kmeans_data_10k.json', 'w') as f:
    json.dump(student_data, f, indent=2)

print("Successfully generated 10,000 student records in 'generated_student_data_10k.json'")