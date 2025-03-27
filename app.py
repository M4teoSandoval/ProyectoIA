import streamlit as st
import pandas as pd
import joblib
import numpy as np


# Cargar los modelos
scaler = joblib.load("scaler.pkl")
svc_model = joblib.load("svc_model.pkl")

# Título y subtítulo
st.title("Modelo predicción de deserción universitaria con IA")
st.subheader("Realizado por Mateo Sandoval, Wilson Suarez, Cristian Cala")

# Introducción
st.write("Esta aplicación utiliza Inteligencia Artificial para predecir si un estudiante continuará o abandonará su carrera universitaria, basándose en diversos factores académicos y personales. ")

# Imagen
st.image("https://www.reporterosasociados.com.co/wp/wp-content/uploads/2023/06/Estudiante-universitaria-en-una-Aula-de-Clase.Foto-Ferran-Nadeu.jpg", use_container_width=True)

# Entrada de datos
st.sidebar.header("Introduce los datos del estudiante")

import streamlit as st
import pandas as pd

# Sidebar con los inputs del usuario
age = st.sidebar.slider("Edad", 16, 25, 20)
mother_education = st.sidebar.selectbox("Nivel Educacion de la Madre", [1, 2, 3, 4], format_func=lambda x: ["Sin estudios", "Primaria", "Bachillerato", "Profesional"][x-1])
father_education = st.sidebar.selectbox("Nivel Educacion de el Padre", [1, 2, 3, 4], format_func=lambda x: ["Sin estudios", "Primaria", "Bachillerato", "Profesional"][x-1])
travel_time = st.sidebar.selectbox("Tiempo de desplazamiento a la Universidad", [1, 2, 3, 4], format_func=lambda x: ["< 15 min", "15-30 min", "30-60 min", "> 1 hora"][x-1])
study_time = st.sidebar.selectbox("tiempo de estudio", [1, 2, 3, 4], format_func=lambda x: ["< 1 hora", "2 horas", "3 horas", "4+ horas"][x-1])
number_of_failures = st.sidebar.selectbox("Numero de materias perdidas", [0, 1, 2, 3])
weekend_alcohol = st.sidebar.selectbox("Consumo de alcohol fines de semana", [1, 2, 3, 4, 5], format_func=lambda x: ["Nunca", "Rara vez", "Moderado", "Frecuente", "Excesivo"][x-1])
weekday_alcohol = st.sidebar.selectbox("Consumo de alcohol entre semana", [1, 2, 3, 4, 5], format_func=lambda x: ["Nunca", "Rara vez", "Moderado", "Frecuente", "Excesivo"][x-1])
health_status = st.sidebar.selectbox("Estado de Salud", [1, 2, 3, 4, 5], format_func=lambda x: ["Muy malo", "Malo", "Regular", "Bueno", "Excelente"][x-1])
number_of_absences = st.sidebar.slider("Numero de Inasistencias", 0, 32, 5)
final_grade = st.sidebar.slider("Nota Promedio ultimo periodo  academico cursado", 0, 5, 3)
address = st.sidebar.selectbox("Tipo de Residencia", ["U", "R"], format_func=lambda x: "Urbana" if x == "U" else "Rural")
parental_status = st.sidebar.selectbox("Estado civil de los Padres", ["T", "A"], format_func=lambda x: "Juntos" if x == "T" else "Divorciados")
school_support = st.sidebar.selectbox("Apoyo Universidad", ["Si", "No"])
family_support = st.sidebar.selectbox("Apoyo Familiar", ["Si", "No"])
internet_access = st.sidebar.selectbox("Acceso a internet", ["Si", "No"])
free_time = st.sidebar.slider("Tiempo Libre", 1, 5, 3)


# Convertir los datos a DataFrame con los nombres correctos
data = pd.DataFrame({
    "Age": [age],
    "Address": [1 if address == "U" else 0],
    "Parental_Status": [1 if parental_status == "T" else 0],
    "Mother_Education": [mother_education],
    "Father_Education": [father_education],
    "Travel_Time": [travel_time],
    "Study_Time": [study_time],
    "Number_of_Failures": [number_of_failures],
    "School_Support": [1 if school_support == "Yes" else 0],
    "Family_Support": [1 if family_support == "Yes" else 0],
    "Internet_Access": [1 if internet_access == "Yes" else 0],
    "Free_Time": [free_time],
    "Weekend_Alcohol_Consumption": [weekend_alcohol],
    "Weekday_Alcohol_Consumption": [weekday_alcohol],
    "Health_Status": [health_status],
    "Number_of_Absences": [number_of_absences],
    "final_grade_converted": [final_grade]
})

# Normalizar los datos
scaled_data = scaler.transform(data)

# Predicción
prediction = svc_model.predict(scaled_data)[0]

# Mostrar el resultado
st.markdown("---")
if prediction:
    st.markdown("<h2 style='color: red; text-align: center;'>❌ Si vas a abandonar tu carrera ❌</h2>", unsafe_allow_html=True)
else:
    st.markdown("<h2 style='color: blue; text-align: center;'>✅ Si vas a continuar con tu carrera ✅</h2>", unsafe_allow_html=True)

# Footer
st.markdown("<p style='text-align: center;'>&copy; Unab2025</p>", unsafe_allow_html=True)
