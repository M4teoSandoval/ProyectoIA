import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Cargar modelos
scaler = joblib.load("/mnt/data/scaler.pkl")
svc_model = joblib.load("/mnt/data/svc_model.pkl")

# Configuración de la aplicación
st.title("Modelo predicción de deserción universitaria con IA")
st.subheader("Realizado por Mateo Sandoval, Wilson Suarez, Cristian Cala")

# Introducción
st.write("Esta aplicación utiliza inteligencia artificial para predecir la deserción universitaria "
         "basándose en diversos factores relacionados con el estudiante y su entorno académico y familiar.")

# Imagen
st.image("https://www.reporterosasociados.com.co/wp/wp-content/uploads/2023/06/Estudiante-universitaria-en-una-Aula-de-Clase.Foto-Ferran-Nadeu.jpg", use_column_width=True)

# Entrada de datos
st.markdown("### Ingrese los datos del estudiante")

age = st.slider("Age", 16, 25, 20)
mother_education = st.selectbox("Mother_Education", [1, 2, 3, 4])
father_education = st.selectbox("Father_Education", [1, 2, 3, 4])
travel_time = st.selectbox("Travel_Time", [1, 2, 3, 4])
study_time = st.selectbox("Study_Time", [1, 2, 3, 4])
number_of_failures = st.selectbox("Number_of_Failures", [0, 1, 2, 3])
weekend_alcohol = st.selectbox("Weekend_Comsumption_Alcohol", [1, 2, 3, 4, 5])
weekday_alcohol = st.selectbox("Weekday_Comsumption_Alcohol", [1, 2, 3, 4, 5])
health_status = st.selectbox("Health_Status", [1, 2, 3, 4, 5])
number_of_absences = st.slider("Number_of_Absences", 0, 32, 5)
final_grade = st.slider("final_grade_converted", 0, 5, 3)
address = st.selectbox("Adress", ["U", "R"])
parental_status = st.selectbox("Parental_Status", ["A", "T"])
school_support = st.selectbox("School_Support", ["Yes", "No"])
family_support = st.selectbox("Family_Support", ["Yes", "No"])
internet_access = st.selectbox("Internet_Access", ["Yes", "No"])
free_time = st.selectbox("Free_Time", [1, 2, 3, 4, 5])

# Creación del dataframe
data = pd.DataFrame({
    "Age": [age],
    "Mother_Education": [mother_education],
    "Father_Education": [father_education],
    "Travel_Time": [travel_time],
    "Study_Time": [study_time],
    "Number_of_Failures": [number_of_failures],
    "Weekend_Comsumption_Alcohol": [weekend_alcohol],
    "Weekday_Comsumption_Alcohol": [weekday_alcohol],
    "Health_Status": [health_status],
    "Number_of_Absences": [number_of_absences],
    "final_grade_converted": [final_grade],
    "Adress": [address],
    "Parental_Status": [parental_status],
    "School_Support": [school_support],
    "Family_Support": [family_support],
    "Internet_Access": [internet_access],
    "Free_Time": [free_time]
})

# Normalizar datos
scaled_data = scaler.transform(data.select_dtypes(include=[np.number]))

# Predicción
prediction = svc_model.predict(scaled_data)[0]

# Mostrar resultado
st.markdown("---")
if prediction:
    st.markdown("<div style='background-color: red; padding: 10px; text-align: center; color: white; font-size: 20px;'>😢 Si vas a abandonar tu carrera</div>", unsafe_allow_html=True)
else:
    st.markdown("<div style='background-color: blue; padding: 10px; text-align: center; color: white; font-size: 20px;'>😊 Si vas a continuar con tu carrera</div>", unsafe_allow_html=True)

# Copyright
st.markdown("<p style='text-align: center;'>&copy; Unab2025</p>", unsafe_allow_html=True)
