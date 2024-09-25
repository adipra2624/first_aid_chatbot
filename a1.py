from flask import Flask, render_template, request, redirect, url_for
from database import get_available_appointments, book_appointment

app = Flask(__name__, template_folder='templates')

@app.route('/')
def index():
    doctors = ['Dr. Smith', 'Dr. Brown', 'Dr. Johnson']
    print("Doctors list:", doctors)  # Debug statement to check doctors list
    return render_template('i1.html', doctors=doctors)  # Ensure this file exists in templates

@app.route('/book', methods=['POST'])
def book():
    doctor = request.form['doctor']
    appointments = get_available_appointments(doctor)
    return render_template('book.html', doctor=doctor, appointments=appointments)

@app.route('/confirm', methods=['POST'])
def confirm():
    doctor = request.form['doctor']
    time = request.form['appointment']
    patient_name = request.form['name']
    book_appointment(doctor, time, patient_name)
    return render_template('confirm.html', doctor=doctor, time=time)

if __name__ == '__main__':
    app.run(debug=True)