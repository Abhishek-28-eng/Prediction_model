# Ensure all subject columns are numeric
new_7th = new_7th.apply(pd.to_numeric, errors='coerce')

# Function to find top and second-top subjects
def find_top_interests(row):
    # Create a dictionary of subjects and their marks
    subject_marks = {subject: mark for subject, mark in zip(row.index, row)}
    
    # Find the highest and second-highest marks
    unique_marks = sorted(set(subject_marks.values()), reverse=True)
    if len(unique_marks) >= 2:
        top_marks = unique_marks[:2]  # Top two unique marks
    elif len(unique_marks) == 1:
        top_marks = unique_marks  # Only one unique mark
    else:
        top_marks = []  # No marks present
    
    # Collect all subjects with marks in top_marks
    top_subjects = [subject for subject, mark in subject_marks.items() if mark in top_marks]
    return ', '.join(top_subjects)

# Apply the function to the single row (axis=1 for row-wise operation)
new_7th['Interest'] = new_7th.apply(find_top_interests, axis=1)
new_7th
# file_path = "Latest7th.csv"  # Specify your desired file path
# new_7th.to_csv(file_path, index=False)  # Save the DataFrame without the index
# print(f"CSV file saved")