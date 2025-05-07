import streamlit as st
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Funciones de carga
@st.cache_resource
def cargar_modelo():
    with open('best_model.pkl', 'rb') as archivo:
        modelo = pickle.load(archivo)
    return modelo

@st.cache_resource
def cargar_scaler():
    with open('scaler.pkl', 'rb') as archivo:
        scaler = pickle.load(archivo)
    return scaler

@st.cache_resource
def cargar_label_encoders():
    with open('label_encoders.pkl', 'rb') as archivo:
        label_encoders_cargado = pickle.load(archivo)
        return label_encoders_cargado

@st.cache_resource
def cargar_train_columns():
    with open('train_columns.pkl', 'rb') as archivo:
        train_cols_cargado = pickle.load(archivo)
        return train_cols_cargado

# Cargar los objetos guardados
model = cargar_modelo()
scaler = cargar_scaler()
label_encoders = cargar_label_encoders()
train_columns = cargar_train_columns()

# Configura la interfaz de usuario en Streamlit
st.title("Predicción de Precio de Vivienda")

st.write("""
    Introduce las características de la casa para predecir su precio.
""")

# Entradas del usuario
area = st.number_input('Área (pies cuadrados)', min_value=100, max_value=10000, value=2500)
bedrooms = st.number_input('Número de Dormitorios', min_value=1, max_value=10, value=3)
bathrooms = st.number_input('Número de Baños', min_value=1, max_value=5, value=2)
stories = st.number_input('Número de Pisos', min_value=1, max_value=4, value=2)
guestroom = st.selectbox('¿Tiene Habitación de Invitados?', list(label_encoders['guestroom'].classes_)) # Usar clases cargadas
hotwaterheating = st.selectbox('¿Tiene Agua Caliente?', list(label_encoders['hotwaterheating'].classes_)) # Usar clases cargadas
airconditioning = st.selectbox('¿Tiene Aire Acondicionado?', list(label_encoders['airconditioning'].classes_)) # Usar clases cargadas
parking = st.number_input('Número de Estacionamientos', min_value=0, max_value=3, value=1)

# Crear el DataFrame para las predicciones (AHORA AQUÍ)
input_data = pd.DataFrame([[area, bedrooms, bathrooms, stories, guestroom, hotwaterheating, airconditioning, parking]],
                          columns=train_columns)

# Codificar las variables categóricas usando los LabelEncoders cargados
for col in ['guestroom', 'hotwaterheating', 'airconditioning']:
    if col in input_data.columns and col in label_encoders:
        if input_data[col][0] not in label_encoders[col].classes_:
            input_data[col] = [label_encoders[col].classes_[0]] # Manejo simple de valores desconocidos
        input_data[col] = label_encoders[col].transform(input_data[col])
print(f"Tipo de input_data antes del escalado: {type(input_data)}")
print(f"Contenido de input_data antes del escalado:\n{input_data}")
print(f"Columnas de input_data antes del escalado: {input_data.columns.tolist()}")
print(f"Columnas esperadas por el scaler (cargadas): {train_columns}")
# Escalar los datos de entrada
input_data_scaled = scaler.transform(input_data)

# Realiza la predicción con el modelo cargado
predicted_price = model.predict(input_data_scaled)[0]

# Muestra el resultado
st.write(f"**Precio estimado de la casa:** ${predicted_price:,.2f}")