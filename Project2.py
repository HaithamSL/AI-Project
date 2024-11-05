#Phase 2 Project

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv('data_week6.csv')


features = ['gender', 'age', 'hypertension', 'heart_disease', 'smoking_history', 'HbA1c_level', 'diabetes']
for feature in features:
    plt.figure(figsize=(8, 4))
    sns.countplot(data=data, x=feature)
    plt.title(f'Distribution of {feature}')
    plt.show()
plt.figure(figsize=(8, 4))
sns.countplot(data=data, x='gender', hue='diabetes')
plt.title('Diabetes vs Gender')
plt.show()

plt.figure(figsize=(8, 4))
sns.countplot(data=data, x='hypertension', hue='diabetes')
plt.title('Diabetes vs Hypertension')
plt.show()

plt.figure(figsize=(8, 4))
sns.countplot(data=data, x='heart_disease', hue='diabetes')
plt.title('Diabetes vs Heart Disease')
plt.show()

plt.figure(figsize=(8, 4))
sns.countplot(data=data, x='smoking_history', hue='diabetes')
plt.title('Diabetes vs Smoking History')
plt.show()

diagnosis_list = []
for level in data['HbA1c_level']:
    if level < 5.7:
        diagnosis_list.append('Normal')
    elif 5.7 <= level <= 6.4:
        diagnosis_list.append('Prediabetes')
    else:
        diagnosis_list.append('Diabetes')

data['initial_diagnosis'] = diagnosis_list
plt.figure(figsize=(8, 4))
sns.countplot(data=data, x='initial_diagnosis', hue='diabetes')
plt.title('Initial Diagnosis vs Diabetes')
plt.show()

data_encoded = pd.get_dummies(data, columns=['gender', 'smoking_history'], drop_first=True)

duplicates_count = data.duplicated().sum()
print(f'Duplicates found: {duplicates_count}')

if duplicates_count > 0:
    data = data.drop_duplicates()

from sklearn.preprocessing import MinMaxScaler

# Selecting numeric features for scaling
numeric_features = ['age', 'BMI', 'HbA1c_level', 'blood_glucose_level']

scaler = MinMaxScaler()
data[numeric_features] = scaler.fit_transform(data[numeric_features])


from sklearn.model_selection import train_test_split

X = data_encoded.drop('diabetes', axis=1)
y = data_encoded['diabetes']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train, y_train)

from sklearn.metrics import accuracy_score

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Model accuracy: {accuracy * 100:.2f}%')


new_patient = {
    'gender_Female': 1, 'age': 48, 'hypertension': 0, 'heart_disease': 1,
    'smoking_history_current': 1, 'BMI': 28.4, 'HbA1c_level': 6.2, 'blood_glucose_level': 120
}

data_encoded = data_encoded.append(new_patient, ignore_index=True)
data_encoded = pd.get_dummies(data_encoded, columns=['gender', 'smoking_history'], drop_first=True)

data_encoded[numeric_features] = scaler.fit_transform(data_encoded[numeric_features])

new_patient_processed = data_encoded.iloc[-1:].drop('diabetes', axis=1)

prediction = model.predict(new_patient_processed)
diabetic_status = 'Diabetic' if prediction[0] == 1 else 'Not Diabetic'
print(f'The patient is {diabetic_status}')

