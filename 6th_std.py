#!/usr/bin/env python
# coding: utf-8

# In[538]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score,classification_report
from sklearn.preprocessing import OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from xgboost import XGBClassifier


# In[539]:


df=pd.read_csv("6th_student_interest_career.csv")
df


# In[540]:


df.dtypes


# In[541]:


df2 = df.drop(['Interest', 'Recommended Career'], axis=1)
df2


# In[542]:


df2['Interest'] = df2.idxmax(axis=1)
df2


# In[543]:


career_map = {
    "Marathi": ["Media and Journalism", "Entertainment and Arts", "Literature and Education", "Government and Administrative Services"],
    "Urdu": ["Media and Journalism", "Entertainment and Arts", "Literature and Education", "Government and Administrative Services"],
    "Hindi": ["Media and Journalism", "Entertainment and Arts", "Literature and Education", "Government and Administrative Services"],
    "English": ["Media and Journalism", "Entertainment and Arts", "Literature and Education", "Government and Administrative Services", "Corporate and Business Communication"],
    "History": ["Historian", "Archaeologist", "Museum Professional", "Cultural Heritage Manager", "Genealogist", "Documentary Filmmaker"],
    "Science": ["Healthcare and Medicine", "Research and Development", "Agriculture and Food Sciences", "Education and Academia", "Allied Healthcare Professions", "Interdisciplinary Fields"],
    "Geography": ["Urban and Regional Planner", "Environmental Consultant", "Cartographer", "Geospatial Analyst", "Meteorologist", "Geologist", "Disaster Management Specialist", "Archaeologist", "Geographic Information Systems (GIS) Specialist", "Geotechnical Engineer"],
    "Drawing": ["Fine Artist", "Graphic Designer", "Concept Artist", "Animator", "Fashion Designer", "Comic Artist/Cartoonist", "Product Designer", "Architect", "Storyboard Artist", "Set Designer/Illustrator for Theatre or Film", "Visual Development Artist", "3D Artist"],
    "Sports": ["Professional Athlete", "Sports Coach", "Sports Nutritionist/Dietitian", "Physical Therapist", "Fitness Trainer"],
    "Environmental Studies": ["Environmental Scientist"],
    "Math": ["Engineering", "Research and Development (R&D)", "Mathematics and Actuarial Science", "Architecture", "Aviation and Aerospace", "Finance and Investment"],
    "Computer": ["Software Developer", "Artificial Intelligence (AI) / Machine Learning Engineer"]
}

# Mapping Recommended Careers and converting to comma-separated strings
df2['Recommended Career'] = df2['Interest'].map(lambda x: ', '.join(career_map.get(x, [])))
df2


# In[544]:


df2.describe()


# In[ ]:






# In[545]:


df2.info


# In[546]:


df2.isnull().sum()


# In[547]:


max_marks = df2['Marathi'].max()

# Count the number of students with the maximum marks
count_max_marks = df2[df2['Marathi'] == max_marks].shape[0]

print(f"Number of students with the maximum marks in Marathi: {count_max_marks}")


# In[548]:


max_marks = df2['Urdu'].max()

# Count the number of students with the maximum marks
count_max_marks = df2[df2['Urdu'] == max_marks].shape[0]

print(f"Number of students with the maximum marks in Urdu: {count_max_marks}")


# In[549]:


max_marks = df2['Hindi'].max()

# Count the number of students with the maximum marks
count_max_marks = df2[df2['Hindi'] == max_marks].shape[0]

print(f"Number of students with the maximum marks in Hindi: {count_max_marks}")


# In[550]:


max_marks = df2['English'].max()

# Count the number of students with the maximum marks
count_max_marks = df2[df2['English'] == max_marks].shape[0]

print(f"Number of students with the maximum marks in English: {count_max_marks}")


# In[551]:


max_marks = df2['History'].max()

# Count the number of students with the maximum marks
count_max_marks = df2[df2['History'] == max_marks].shape[0]

print(f"Number of students with the maximum marks in History: {count_max_marks}")


# In[552]:


max_marks = df2['Science'].max()

# Count the number of students with the maximum marks
count_max_marks = df2[df2['Science'] == max_marks].shape[0]

print(f"Number of students with the maximum marks in Science: {count_max_marks}")


# In[553]:


max_marks = df2['Geography'].max()

# Count the number of students with the maximum marks
count_max_marks = df2[df2['Geography'] == max_marks].shape[0]

print(f"Number of students with the maximum marks in Geography: {count_max_marks}")


# In[554]:


max_marks = df2['Drawing'].max()

# Count the number of students with the maximum marks
count_max_marks = df2[df2['Drawing'] == max_marks].shape[0]

print(f"Number of students with the maximum marks in Drawing: {count_max_marks}")


# In[555]:


max_marks = df2['Sports'].max()

# Count the number of students with the maximum marks
count_max_marks = df2[df2['Sports'] == max_marks].shape[0]

print(f"Number of students with the maximum marks in Sports: {count_max_marks}")


# In[556]:


max_marks = df2['Environmental Studies'].max()

# Count the number of students with the maximum marks
count_max_marks = df2[df2['Environmental Studies'] == max_marks].shape[0]

print(f"Number of students with the maximum marks in Environmental Studies: {count_max_marks}")


# In[557]:


max_marks = df2['Computer'].max()

# Count the number of students with the maximum marks
count_max_marks = df2[df2['Computer'] == max_marks].shape[0]

print(f"Number of students with the maximum marks in Computer: {count_max_marks}")


# In[558]:


max_marks = df2['Math'].max()

# Count the number of students with the maximum marks
count_max_marks = df2[df2['Math'] == max_marks].shape[0]

print(f"Number of students with the maximum marks in Math: {count_max_marks}")


# In[559]:


#x = df.drop(['Recommended Career'], axis=1)


# In[560]:


#x=x.values


# In[561]:


# # Extract the subject of interest based on max marks
# subjects = ["Marathi", "Urdu", "Hindi", "English", "History", "Science", 
#             "Geography", "Drawing", "Sports", "Environmental Studies", "Math", "Computer"]
# df['Interest'] = df[subjects].idxmax(axis=1)  # Identify the subject with max marks

# # One-hot encode the 'Interest' column
# encoder = OneHotEncoder()
# interest_encoded = encoder.fit_transform(df[['Interest']]).toarray()


# In[562]:


# Elbow method to find the optimal number of clusters
# wcss = []
# for i in range(1, 20):  # Test 1 to 10 clusters
#     kmeans = KMeans(n_clusters=i, random_state=42)
#     kmeans.fit(interest_encoded)
#     wcss.append(kmeans.inertia_)

# # Plot the Elbow Curve
# plt.figure(figsize=(10, 6))
# plt.plot(range(1, 20), wcss, marker='o', linestyle='--')
# plt.title('Elbow Method for Optimal Clusters')
# plt.xlabel('Number of Clusters')
# plt.ylabel('WCSS')
# plt.xticks(range(1, 20))
# plt.grid()
# plt.show()


# In[563]:


# optimal_clusters = 12  # Replace with the actual number determined from the elbow curve
# kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
# y_means['Cluster']= kmeans.fit_predict(interest_encoded)

# # Analyze and save results
# print(df[['Interest', 'Cluster']].head(12))
# df.to_csv("clustered_stu.csv", index=False)


# In[564]:


# a=df['Recommended Career']
# y_means = pd.DataFrame(y_means)
# #Converts the NumPy array y_means (containing cluster assignments) into a pandas DataFrame.

# z = pd.concat([y_means, a],axis = 1)
# #Concatenates the two DataFrames (y_means and a) along the columns (axis=1), combining the cluster assignments with the 
#     #original labels.

# z = z.rename(columns = {0: 'cluster'})


# In[565]:


# # Group the data by clusters
# clusters = df.groupby('Cluster')

# # View data for each cluster
# for cluster_id, cluster_data in clusters:
#     print(f"Cluster {cluster_id}:\n")
#     print(cluster_data.head())  # Display the first few rows of the cluster
#     print("\n" + "-"*50 + "\n")


# In[566]:


#x = df.drop(['Recommended Career'], axis=1)


# In[567]:


# Features and Target
X = df2[['Interest']]
y = df2['Recommended Career']


# In[ ]:





# In[568]:


le = LabelEncoder()
df2['Recommended Career'] = le.fit_transform(df2['Recommended Career'])  # Fit label encoder on the entire dataset
y = df2['Recommended Career']


# In[569]:


X = pd.get_dummies(X)


# In[570]:


# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[571]:


ensemble_model = VotingClassifier(estimators=[('rf', RandomForestClassifier()), ('xgb', XGBClassifier())], voting='soft')
ensemble_model.fit(X_train, y_train)


# In[572]:


# Train a model (Random Forest in this case)
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)


# In[573]:


# Predictions
y_pred = model.predict(X_test)


# In[574]:


# Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))


# In[599]:


# Test with a new student
new_student = pd.DataFrame({
    'Interest': ['Math']
   
    
})
new_student = pd.get_dummies(new_student)
new_student = new_student.reindex(columns=X.columns, fill_value=0)


# In[600]:


predicted_career = model.predict(new_student)
predicted_output=le.inverse_transform(predicted_career)
print("Predicted Career:",predicted_output)


# In[ ]:





# In[ ]:




