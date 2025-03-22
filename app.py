import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

# Cargar modelos
scaler = joblib.load("scaler.pkl")
svc_model = joblib.load("svc_model.pkl")

# Configurar la página
st.title("Modelo predicción de deserción universitaria con IA")
st.subheader("Realizado por Mateo Sandoval, Wilson Suarez, Cristian Cala")

# Introducción
st.write("Esta aplicación utiliza un modelo de Machine Learning para predecir si un estudiante tiene riesgo de desertar de la universidad.")

# Imagen
st.image("https://www.reporterosasociados.com.co/wp/wp-content/uploads/2023/06/Estudiante-universitaria-en-una-Aula-de-Clase.Foto-Ferran-Nadeu.jpg")

# Entrada de datos
age = st.slider("Age", 16, 25, 20)
mother_education = st.selectbox("Mother_Education", [1, 2, 3, 4])
father_education = st.selectbox("Father_Education", [1, 2, 3, 4])
travel_time = st.selectbox("Travel_Time", [1, 2, 3, 4])
study_time = st.selectbox("Study_Time", [1, 2, 3, 4])
number_of_failures = st.selectbox("Number_of_Failures", [0, 1, 2, 3])
weekend_consumption = st.selectbox("Weekend_Comsumption_Alcohol", [1, 2, 3, 4, 5])
weekday_consumption = st.selectbox("Weekday_Comsumption_Alcohol", [1, 2, 3, 4, 5])
health_status = st.selectbox("Health_Status", [1, 2, 3, 4, 5])
number_of_absences = st.slider("Number_of_Absences", 0, 32, 5)
final_grade_converted = st.slider("final_grade_converted", 0, 5, 3)
address = st.selectbox("Adress", ["U", "R"])
parental_status = st.selectbox("Parental_Status", ["A", "T"])
school_support = st.selectbox("School_Support", ["Yes", "No"])
family_support = st.selectbox("Family_Support", ["Yes", "No"])
internet_access = st.selectbox("Internet_Access", ["Yes", "No"])
free_time = st.slider("Free_Time", 1, 5, 3)

# Convertir datos a DataFrame
data = pd.DataFrame([[age, mother_education, father_education, travel_time, study_time, number_of_failures,
                      weekend_consumption, weekday_consumption, health_status, number_of_absences,
                      final_grade_converted, address, parental_status, school_support, family_support,
                      internet_access, free_time]],
                    columns=["Age", "Mother_Education", "Father_Education", "Travel_Time", "Study_Time", "Number_of_Failures",
                             "Weekend_Comsumption_Alcohol", "Weekday_Comsumption_Alcohol", "Health_Status", "Number_of_Absences",
                             "final_grade_converted", "Adress", "Parental_Status", "School_Support", "Family_Support",
                             "Internet_Access", "Free_Time"])

# Normalizar datos
numeric_cols = ["Age", "Mother_Education", "Father_Education", "Travel_Time", "Study_Time", "Number_of_Failures",
                "Weekend_Comsumption_Alcohol", "Weekday_Comsumption_Alcohol", "Health_Status", "Number_of_Absences",
                "final_grade_converted", "Free_Time"]
data[numeric_cols] = scaler.transform(data[numeric_cols])

# Predecir
y_pred = svc_model.predict(data)[0]

# Mostrar resultado
if y_pred:
    st.markdown("<div style='background-color:red; padding:10px; color:white; text-align:center;'>🚨 Si vas a abandonar tu carrera 🚨</div>", unsafe_allow_html=True)
else:
    st.markdown("<div style='background-color:blue; padding:10px; color:white; text-align:center;'>🎓 Si vas a continuar con tu carrera 🎓</div>", unsafe_allow_html=True)

# Línea divisoria
st.markdown("---")

# Copyright
st.markdown("<p style='text-align:center;'>&copy; Unab2025</p>", unsafe_allow_html=True)
