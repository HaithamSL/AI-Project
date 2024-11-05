#Phase 1:
    
import pandas as pd


data = pd.read_csv('data_week6.csv')
print(data.columns)

data.head()
data.info()
gender_distribution = data['gender'].value_counts()
print(gender_distribution)

smoking_history_distribution = data['smoking_history'].value_counts()
print(smoking_history_distribution)

correlation = data[['diabetes', 'age', 'BMI', 'blood_glucose_level']].corr()
print(correlation)

highest_correlation = correlation.unstack().sort_values(ascending=False)
print(highest_correlation)

over_50_count = data[data['age'] > 50].shape[0]
print(over_50_count)

male_diabetic_count = data[(data['gender'] == 'Male') & (data['diabetes'] == 1)].shape[0]
print(male_diabetic_count)

female_non_smoker_diabetic_count = data[(data['gender'] == 'Female') & 
                                        (data['smoking_history'] == 'never') & 
                                        (data['diabetes'] == 1)].shape[0]
print(female_non_smoker_diabetic_count)

oldest_heart_disease_patient = data[data['heart_disease'] == 1]['age'].max()
print(oldest_heart_disease_patient)

hypertension_percentage = (data[data['hypertension'] == 1].shape[0] / data.shape[0]) * 100
print(hypertension_percentage)

mean_age_by_diabetes_gender = data.groupby(['diabetes', 'gender'])['age'].mean()
print(mean_age_by_diabetes_gender)

def search_probability(smoking_history, gender):
    total_count = data[(data['smoking_history'] == smoking_history) & (data['gender'] == gender)].shape[0]
    diabetic_count = data[(data['smoking_history'] == smoking_history) & 
                          (data['gender'] == gender) & 
                          (data['diabetes'] == 1)].shape[0]
    
    probability = diabetic_count / total_count if total_count > 0 else 0
    return probability

print(search_probability('never', 'Male'))

def categorize_and_count(data):
    age_groups = {'Young': 0, 'Middle-aged': 0, 'Senior': 0}
    
    for index, row in data.iterrows():
        if row['age'] < 30:
            age_group = 'Young'
        elif 30 <= row['age'] <= 60:
            age_group = 'Middle-aged'
        else:
            age_group = 'Senior'
        
        age_groups[age_group] += 1

    return age_groups

age_group_counts = categorize_and_count(data)
print(age_group_counts)


