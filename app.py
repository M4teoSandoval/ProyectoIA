import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Cargar los modelos
scaler = joblib.load("scaler.pkl")
svc_model = joblib.load("svc_model.pkl")

# Título y subtítulo
st.title("Modelo de Predicción de Deserción Universitaria con IA")
st.subheader("Realizado por Mateo Sandoval, Wilson Suarez, Cristian Cala")

# Introducción
st.write("Esta aplicación utiliza Inteligencia Artificial para predecir si un estudiante continuará o abandonará su carrera universitaria, basándose en diversos factores académicos y personales.")

# Imagen
st.image("https://www.reporterosasociados.com.co/wp/wp-content/uploads/2023/06/Estudiante-universitaria-en-una-Aula-de-Clase.Foto-Ferran-Nadeu.jpg", use_container_width=True)

# Entrada de datos
st.sidebar.header("Introduce los datos del estudiante")

Age = st.sidebar.slider("Edad", 16, 25, 20)
Mother_Education = st.sidebar.selectbox("Educación de la Madre", [1, 2, 3, 4], format_func=lambda x: ["Sin estudios", "Primaria", "Bachillerato", "Profesional"][x-1])
Father_Education = st.sidebar.selectbox("Educación del Padre", [1, 2, 3, 4], format_func=lambda x: ["Sin estudios", "Primaria", "Bachillerato", "Profesional"][x-1])
Travel_Time = st.sidebar.selectbox("Tiempo de Viaje", [1, 2, 3, 4], format_func=lambda x: ["< 15 min", "15-30 min", "30-60 min", "> 1 hora"][x-1])
Study_Time = st.sidebar.selectbox("Tiempo de Estudio", [1, 2, 3, 4], format_func=lambda x: ["< 1 hora", "2 horas", "3 horas", "4+ horas"][x-1])
Number_of_Failures = st.sidebar.selectbox("Número de Fracasos Académicos", [0, 1, 2, 3])
Weekend_Alcohol = st.sidebar.selectbox("Consumo de Alcohol en Fin de Semana", [1, 2, 3, 4, 5], format_func=lambda x: ["Nunca", "Rara vez", "Moderado", "Frecuente", "Excesivo"][x-1])
Weekday_Alcohol = st.sidebar.selectbox("Consumo de Alcohol entre Semana", [1, 2, 3, 4, 5], format_func=lambda x: ["Nunca", "Rara vez", "Moderado", "Frecuente", "Excesivo"][x-1])
Health_Status = st.sidebar.selectbox("Estado de Salud", [1, 2, 3, 4, 5], format_func=lambda x: ["Muy malo", "Malo", "Regular", "Bueno", "Excelente"][x-1])
Number_of_Absences = st.sidebar.slider("Número de Ausencias", 0, 32, 5)
Final_Grade = st.sidebar.slider("Nota Final Convertida", 0, 5, 3)
Free_Time = st.sidebar.slider("Tiempo Libre", 1, 5, 3)

# Convertir los datos a DataFrame
data = pd.DataFrame({
    "Age": [Age],
    "Mother_Education": [Mother_Education],
    "Father_Education": [Father_Education],
    "Travel_Time": [Travel_Time],
    "Study_Time": [Study_Time],
    "Number_of_Failures": [Number_of_Failures],
    "Weekend_Alcohol_Consumption": [Weekend_Alcohol],
    "Weekday_Alcohol_Consumption": [Weekday_Alcohol],
    "Health_Status": [Health_Status],
    "Number_of_Absences": [Number_of_Absences],
    "final_grade_converted": [Final_Grade],
    "Free_Time": [Free_Time]
})

# Verificar que los datos tengan las columnas correctas
expected_features = ['Age', 'Mother_Education', 'Father_Education', 'Travel_Time', 'Study_Time',
                     'Number_of_Failures', 'Free_Time', 'Weekend_Alcohol_Consumption',
                     'Weekday_Alcohol_Consumption', 'Health_Status', 'Number_of_Absences',
                     'final_grade_converted']

# Filtrar las columnas correctas
data = data[expected_features]

# Convertir a valores numéricos para evitar errores
data = data.apply(pd.to_numeric, errors='coerce')

# Verificar si hay valores nulos
if data.isnull().sum().sum() > 0:
    st.error("⚠️ Error en los datos de entrada. Revisa los valores ingresados.")

else:
    # Normalizar los datos
    scaled_data = scaler.transform(data)

    # Predicción
    prediction = svc_model.predict(scaled_data)[0]

    # Mostrar el resultado
    st.markdown("---")
    if prediction == 1:
        st.markdown("<h2 style='color: red; text-align: center;'>❌ Existe un alto riesgo de deserción ❌</h2>", unsafe_allow_html=True)
    else:
        st.markdown("<h2 style='color: blue; text-align: center;'>✅ Probabilidad alta de continuidad ✅</h2>", unsafe_allow_html=True)

# Footer
st.markdown("<p style='text-align: center;'>&copy; Unab2025</p>", unsafe_allow_html=True)
