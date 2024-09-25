appointments_db = {
    'Dr. Smith': ['10:00 AM', '11:00 AM', '2:00 PM'],
    'Dr. Brown': ['9:00 AM', '12:00 PM', '4:00 PM'],
    'Dr. Johnson': ['8:00 AM', '1:00 PM', '3:00 PM']
}

def get_available_appointments(doctor):
    return appointments_db.get(doctor, [])

def book_appointment(doctor, time, patient_name):
    if time in appointments_db[doctor]:
        appointments_db[doctor].remove(time)
        print(f"Appointment booked for {patient_name} with {doctor} at {time}")