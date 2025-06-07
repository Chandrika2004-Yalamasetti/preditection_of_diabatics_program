import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib



# Load the dataset
df = pd.read_csv(r"C:\Users\yelam\OneDrive\Desktop\prediabetes_modified_(3)(1).csv")

# Display the first few rows of the dataframe
print(df.head())

# Separate features and target variable
X = df.drop(columns=['Outcome'])  # Replace 'Outcome' with the name of your target variable
y = df['Outcome']  # Replace 'Outcome' with the name of your target variable

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features (optional, but recommended for some models)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create the XGBoost classifier
model = XGBClassifier()

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print('Confusion Matrix:')
print(conf_matrix)
print('Classification Report:')
print(class_report)

# Save the model to a file (optional)
joblib.dump(model, 'prediabetes_model.pkl')

# Load the model (if needed)
# model = joblib.load('prediabetes_model.pkl')

# Get patient details and predict
patient_df = get_patient_details()
patient_scaled = scaler.transform(patient_df)
patient_prediction = model.predict(patient_scaled)

if patient_prediction[0] == 1:
    print("The patient is likely to have prediabetes.")
else:
    print("The patient is not likely to have prediabetes.")