from flask import Flask, request, render_template
import pandas as pd

app = Flask(__name__, template_folder='backend/templates')

# Load the dataset
data = pd.read_csv('Disease_symptom_and_patient_profile_dataset.csv')

@app.route('/')
def home():
    return render_template('index1.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get input values from the form
    fever = request.form.get('fever')
    cough = request.form.get('cough')
    fatigue = request.form.get('fatigue')
    difficulty_breathing = request.form.get('difficulty_breathing')
    age = int(request.form.get('age'))
    gender = request.form.get('gender')
    blood_pressure = request.form.get('blood_pressure')
    cholesterol_level = request.form.get('cholesterol_level')

    # Create a DataFrame for the input values
    input_data = pd.DataFrame({
        'Fever': [fever],
        'Cough': [cough],
        'Fatigue': [fatigue],
        'Difficulty Breathing': [difficulty_breathing],
        'Age': [age],
        'Gender': [gender],
        'Blood Pressure': [blood_pressure],
        'Cholesterol Level': [cholesterol_level]
    })

    # Check for matching disease
    matching_diseases = data[(data['Fever'] == fever) & 
                              (data['Cough'] == cough) & 
                              (data['Fatigue'] == fatigue) & 
                              (data['Difficulty Breathing'] == difficulty_breathing) & 
                              (data['Age'] == age) & 
                              (data['Gender'] == gender) & 
                              (data['Blood Pressure'] == blood_pressure) & 
                              (data['Cholesterol Level'] == cholesterol_level)]

    # Prepare result
    if not matching_diseases.empty:
        diseases = matching_diseases['Disease'].unique()
        result = ', '.join(diseases)
    else:
        result = "No matching disease found."

    return render_template('result1.html', prediction=result)

if __name__ == "__main__":
    app.run(debug=True)
