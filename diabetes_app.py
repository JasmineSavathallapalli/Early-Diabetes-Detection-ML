import tkinter as tk
from tkinter import messagebox
import joblib
import numpy as np


model = joblib.load('final_knn_model.pkl')
scaler = joblib.load('scaler.pkl')


def predict_diabetes():
    try:
        # Get values from entry fields
        data = [float(entry.get()) for entry in entries]
        scaled_data = scaler.transform([data])
        prediction = model.predict(scaled_data)[0]
        prob = model.predict_proba(scaled_data)[0][1]

        result = "Diabetic" if prediction == 1 else "Not Diabetic"
        messagebox.showinfo("Prediction", f"🧠 Prediction: {result}\n📊 Chance: {prob * 100:.2f}%")
    except:
        messagebox.showerror("Error", "Please enter valid numeric values in all fields.")

app = tk.Tk()
app.title("Diabetes Prediction App")
app.geometry("400x500")


features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
            'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

entries = []


for idx, feature in enumerate(features):
    label = tk.Label(app, text=feature)
    label.pack(pady=(10, 0))
    entry = tk.Entry(app)
    entry.pack()
    entries.append(entry)


predict_btn = tk.Button(app, text="Predict", command=predict_diabetes, bg="blue", fg="white", height=2)
predict_btn.pack(pady=20)


app.mainloop()
