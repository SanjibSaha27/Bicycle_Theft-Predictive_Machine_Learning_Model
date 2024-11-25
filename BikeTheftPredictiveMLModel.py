# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Load the dataset
file_path = 'C:/Sanjib/McMaster University/My Work/Bicycle Theft Predictive Machine Learning Model/Bicycle_Thefts.csv'
bicycle_thefts_data = pd.read_csv(file_path, low_memory=False)

# Handle Missing Values
bicycle_thefts_data['Bike_Make'] = bicycle_thefts_data['Bike_Make'].fillna('Unknown')
bicycle_thefts_data['Bike_Colour'] = bicycle_thefts_data['Bike_Colour'].fillna('Unknown')
bicycle_thefts_data['Cost_of_Bike'] = bicycle_thefts_data['Cost_of_Bike'].fillna(bicycle_thefts_data['Cost_of_Bike'].median())
bicycle_thefts_data.drop(columns=['Bike_Model'], inplace=True, errors='ignore')

# Convert Date Columns to Datetime
bicycle_thefts_data['Occurrence_Date'] = pd.to_datetime(bicycle_thefts_data['Occurrence_Date'], errors='coerce')
bicycle_thefts_data['Report_Date'] = pd.to_datetime(bicycle_thefts_data['Report_Date'], errors='coerce')

# Create Time_to_Report Feature
bicycle_thefts_data['Time_to_Report'] = (bicycle_thefts_data['Report_Date'] - bicycle_thefts_data['Occurrence_Date']).dt.days

# Drop rows with NaN in 'Time_to_Report'
bicycle_thefts_data = bicycle_thefts_data[bicycle_thefts_data['Time_to_Report'].notnull()]

# Encode Non-Numeric Columns
non_numeric_columns = ['Occurrence_DayOfWeek', 'Report_Month', 'Report_DayOfWeek',
                       'Division', 'City', 'Hood_ID', 'NeighbourhoodName',
                       'Location_Type', 'Premises_Type', 'Bike_Make', 'Bike_Colour']
bicycle_thefts_data = pd.get_dummies(bicycle_thefts_data, columns=non_numeric_columns, drop_first=True)

# Encode Target Variable
X = bicycle_thefts_data.drop(columns=['Status', 'event_unique_id', 'Occurrence_Date', 'Report_Date'], errors='ignore')
y = bicycle_thefts_data['Status']

# Ensure all features are numeric
X = X.select_dtypes(include=[np.number])

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train Random Forest Model
model = RandomForestClassifier(random_state=42, n_estimators=100)
model.fit(X_train, y_train)

# Evaluate the Model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Display Evaluation Metrics
print(f"Accuracy: {accuracy:.2f}")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Visualize Feature Importance
feature_importances = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 8))
sns.barplot(x='Importance', y='Feature', data=feature_importances, palette='coolwarm')
plt.title('Feature Importance')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()

# Save Cleaned Dataset
cleaned_file_path = 'C:/Sanjib/McMaster University/My Work/Bicycle Theft Predictive Machine Learning Model/Bicycle_Thefts_Cleaned.csv'
bicycle_thefts_data.to_csv(cleaned_file_path, index=False)
print(f"Cleaned dataset saved to {cleaned_file_path}")


