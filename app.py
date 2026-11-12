import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import plotly.graph_objects as go

# ---------------------------------------------------------
# Carga y preparación de datos del archivo Excel real
# ---------------------------------------------------------
@st.cache_data
def load_and_prepare_data():
    file_path = "datos_estudiantes.csv"
    df = pd.read_excel(file_path)
    
    # Nombres de columnas limpios basados en la encuesta UAP
    col_grade_past = "What was your grade the last period? (Specify) Example: 6.7"
    col_grade_current = "What is your grade in the current period? (Approximately)"
    col_study_hours = "How many hours do you study outside class?"
    col_sleep = "How many hours do you sleep per night? "
    col_social = "How many hours per day do you spend on social media or entertainment?"
    col_stress = "How would you rate your stress level during midterm exams?"
    
    # Filtrar valores nulos de la variable objetivo (calificación pasada)
    df = df.dropna(subset=[col_grade_past])
    
    # Limpiar y convertir horas de estudio a numérico (manejo de rangos o strings)
    df["study_hours_num"] = pd.to_numeric(df[col_study_hours], errors="coerce").fillna(3)
    df["grade_past"] = df[col_grade_past]
    df["grade_current"] = pd.to_numeric(df[col_grade_current], errors="coerce").fillna(8)
    df["sleep_hours"] = pd.to_numeric(df[col_sleep], errors="coerce").fillna(7)
    df["social_media"] = pd.to_numeric(df[col_social], errors="coerce").fillna(3)
    df["stress_level"] = pd.to_numeric(df[col_stress], errors="coerce").fillna(5)
    
    # Feature engineering adicional para enriquecer el modelo
    df["eficiencia_estudio"] = df["grade_past"] / (df["study_hours_num"] + 1)
    df["indice_descanso"] = df["sleep_hours"] - (df["stress_level"] / 10)
    
    return df

df = load_and_prepare_data()

# Selección de características (features) para entrenar los modelos
feature_cols = [
    "study_hours_num",
    "grade_current",
    "sleep_hours",
    "social_media",
    "stress_level",
    "eficiencia_estudio",
    "indice_descanso"
]

X = df[feature_cols]
Y = df["grade_past"]  # Objetivo: predecir la calificación del periodo anterior

# Normalización de los datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Entrenamiento de Modelos: Ridge Regression y Random Forest
ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_scaled, Y)

rf_model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=5)
rf_model.fit(X_scaled, Y)

# ---------------------------------------------------------
# Interfaz de Usuario (UI) en Streamlit
# ---------------------------------------------------------
st.title("Predictor de Rendimiento Académico - UAP")
st.markdown("Modelo predictivo basado en hábitos de estudio, descanso y calificaciones actuales.")

# Selector de modelo en la barra lateral
st.sidebar.header("Configuración del Modelo")
model_choice = st.sidebar.selectbox("Selecciona el algoritmo de predicción", ["Random Forest", "Regresión Ridge"])

col1, col2 = st.columns(2)

with col1:
    st.subheader("Hábitos de Estudio")
    study_input = st.number_input("Horas de estudio fuera de clase (semanales/diarias)", min_value=1.0, max_value=30.0, value=5.0, step=0.5)
    grade_current_input = st.number_input("Calificación actual aproximada", min_value=6.0, max_value=10.0, value=9.0, step=0.1)
    social_input = st.number_input("Horas en redes sociales / entretenimiento al día", min_value=0.0, max_value=12.0, value=3.0, step=0.5)

with col2:
    st.subheader("Bienestar y Estrés")
    sleep_input = st.number_input("Horas de sueño por noche", min_value=3.0, max_value=12.0, value=7.0, step=0.5)
    stress_input = st.slider("Nivel de estrés durante exámenes (1-10)", min_value=1, max_value=10, value=5)

# Cálculo de variables derivadas para la predicción en tiempo real
eficiencia_calc = grade_current_input / (study_input + 1)
descanso_calc = sleep_input - (stress_input / 10)

if st.button("Predecir Calificación", type="primary"):
    # Construir dataframe con la misma estructura
    input_data = pd.DataFrame({
        "study_hours_num": [study_input],
        "grade_current": [grade_current_input],
        "sleep_hours": [sleep_input],
        "social_media": [social_input],
        "stress_level": [stress_input],
        "eficiencia_estudio": [eficiencia_calc],
        "indice_descanso": [descanso_calc]
    })
    
    input_scaled = scaler.transform(input_data)
    
    # Seleccionar predicción según el modelo elegido
    if model_choice == "Random Forest":
        prediction = rf_model.predict(input_scaled)[0]
        importances = rf_model.feature_importances_
    else:
        prediction = ridge_model.predict(input_scaled)[0]
        importances = np.abs(ridge_model.coef_)
    
    # Resultados visuales
    st.markdown("---")
    st.subheader("Resultado de la Predicción")
    
    col_res1, col_res2, col_res3 = st.columns(3)
    
    with col_res1:
        st.metric("Calificación Estimada", f"{prediction:.2f}")
    with col_res2:
        st.metric("Eficiencia Calculada", f"{eficiencia_calc:.2f}")
    with col_res3:
        status = "Alto Rendimiento" if prediction >= 9.0 else "Rendimiento Moderado"
        st.metric("Categoría", status)
    
    # Gráfico de indicador visual con Plotly
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = prediction,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Estimación de Calificación (Escala 6-10)"},
        gauge = {
            'axis': {'range': [6.0, 10.0]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [6.0, 7.5], 'color': "lightgray"},
                {'range': [7.5, 9.0], 'color': "lightyellow"},
                {'range': [9.0, 10.0], 'color': "lightgreen"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 9.2
            }
        }
    ))
    st.plotly_chart(fig, use_container_width=True)
    
    # Importancia de características
    st.subheader("Factores Clave del Modelo")
    feature_importance_df = pd.DataFrame({
        'Factor': ['Horas de Estudio', 'Calificación Actual', 'Horas de Sueño', 'Redes Sociales', 'Nivel de Estrés', 'Eficiencia de Estudio', 'Índice de Descanso'],
        'Importancia': importances
    }).sort_values('Importancia', ascending=False)
    
    fig_bar = go.Figure(go.Bar(
        x=feature_importance_df['Importancia'],
        y=feature_importance_df['Factor'],
        orientation='h',
        marker=dict(color='steelblue')
    ))
    fig_bar.update_layout(xaxis_title="Peso / Importancia", height=300)
    st.plotly_chart(fig_bar, use_container_width=True)

# Sección expandible con estadísticas generales del dataset
with st.expander("Ver estadísticas generales del dataset UAP"):
    st.write(f"Total de registros analizados: {len(df)}")
    st.metric("Promedio general de calificaciones pasadas", f"{df['grade_past'].mean():.2f}")
    st.metric("Promedio de horas de estudio", f"{df['study_hours_num'].mean():.2f} hrs")