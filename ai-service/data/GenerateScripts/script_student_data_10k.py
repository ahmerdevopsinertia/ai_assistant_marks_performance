import json
import random
from faker import Faker
from collections import defaultdict

# Initialize faker for random names
fake = Faker()


req_num_students = 100000;
def generate_student_data(num_students=req_num_students):
    # Base structure
    data = {
        "academic_year_2024": {
            "thresholds": {
                "under_performing": 60,
                "subject_thresholds": {"math": 55, "science": 65},
                "min_critical_subjects": 1,
                "use_ml_suggestions": True,
                "ml_score_range": [20, 80],
            },
            "students": {},
        }
    }

    # Generate student records
    for i in range(num_students):
        # Create realistic score distributions
        if i < 100:  # Top performers (1%)
            math_scores = [random.randint(85, 100) for _ in range(3)]
            science_scores = [random.randint(85, 100) for _ in range(3)]
        elif i < 500:  # Good performers (4%)
            math_scores = [random.randint(70, 89) for _ in range(3)]
            science_scores = [random.randint(70, 89) for _ in range(3)]
        elif i < 2000:  # Average performers (15%)
            math_scores = [random.randint(55, 75) for _ in range(3)]
            science_scores = [random.randint(55, 75) for _ in range(3)]
        elif i < 3000:  # Below average (10%)
            math_scores = [random.randint(40, 60) for _ in range(3)]
            science_scores = [random.randint(40, 60) for _ in range(3)]
        elif i < 4000:  # Struggling students (10%)
            math_scores = [random.randint(30, 50) for _ in range(3)]
            science_scores = [random.randint(30, 50) for _ in range(3)]
        else:  # Under-performers (60%)
            math_scores = [random.randint(1, 40) for _ in range(3)]
            science_scores = [random.randint(1, 40) for _ in range(3)]

        # Add slight progression trend (improving/declining/stable)
        trend = random.choice(["improving", "declining", "stable"])
        if trend == "improving":
            math_scores.sort()
            science_scores.sort()
        elif trend == "declining":
            math_scores.sort(reverse=True)
            science_scores.sort(reverse=True)

        # Generate unique username
        username = fake.user_name()
        while username in data["academic_year_2024"]["students"]:
            username = fake.user_name()

        # Add student record
        data["academic_year_2024"]["students"][username] = {
            "subjects": {"math": math_scores, "science": science_scores}
        }

    return data


# Generate the data
student_data = generate_student_data()

# Save to file
with open("generated_student_data_10.json", "w") as f:
    json.dump(student_data, f, indent=2)

print(f"Generated {req_num_students} students records in generated_student_data_10.json")
