import pandas as pd

# Load the CSV files for 5th and 6th standard students
csv_6th = "Datasets/New6th.csv"  # Replace with the actual file name for 6th std
csv_5th = "Datasets/New5th.csv"  # Replace with the actual file name for 5th std

# Read the CSV files into DataFrames
data_6th = pd.read_csv(csv_6th)
data_5th = pd.read_csv(csv_5th)

# Function to find the highest marks and corresponding subjects
def find_highest_subjects(row):
    numeric_row = row.apply(pd.to_numeric, errors='coerce')  # Convert to numeric, ignoring errors
    max_marks = numeric_row.max()
    return numeric_row[numeric_row == max_marks].index.tolist()

# Apply the function to numeric columns only
data_6th['HighestSubjects'] = data_6th.select_dtypes(include='number').apply(find_highest_subjects, axis=1)
data_5th['HighestSubjects'] = data_5th.select_dtypes(include='number').apply(find_highest_subjects, axis=1)

# Function to determine interest based on highest marks
def determine_interest(subjects_6th, subjects_5th):
    subjects_6th_set = set(subjects_6th)
    subjects_5th_set = set(subjects_5th)
    common_subjects = subjects_6th_set.intersection(subjects_5th_set)

    if common_subjects:
        return list(common_subjects)
    else:
        return list(subjects_6th_set.union(subjects_5th_set))

# Determine interests
interests = []
for idx in range(len(data_6th)):
    subjects_6th = data_6th.iloc[idx]['HighestSubjects']
    subjects_5th = data_5th.iloc[idx]['HighestSubjects']
    interests.append(determine_interest(subjects_6th, subjects_5th))

# Add interests to the 6th standard data
data_6th['Interest'] = interests

# Save the result
data_6th[['Interest']].to_csv("student_interests.csv", index=False)

# Display the result
print(data_6th[['Interest']])
