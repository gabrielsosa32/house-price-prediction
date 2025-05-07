:

🏠 House Price Prediction - Machine Learning & Streamlit App

Este proyecto tiene como objetivo predecir el precio de viviendas basándose en diversas características (área, número de habitaciones, baños, pisos, etc.) mediante la aplicación de varios modelos de regresión de machine learning. Inicialmente concebido como parte de mi portfolio de ciencia de datos, el proyecto ha evolucionado para incluir modelos más avanzados y una interfaz de usuario interactiva creada con Streamlit.

📌 Dataset

El conjunto de datos utilizado contiene las siguientes columnas relevantes para la predicción del precio:

price: Precio de la vivienda en moneda local.
area: Superficie de la vivienda en pies cuadrados.
bedrooms: Número de dormitorios.
bathrooms: Número de baños.
stories: Número de pisos de la vivienda.
parking: Número de espacios de estacionamiento.
guestroom: Indica si la vivienda tiene una habitación de invitados (booleano).
hotwaterheating: Indica si la vivienda tiene calefacción de agua caliente (booleano).
airconditioning: Indica si la vivienda tiene aire acondicionado (booleano).
⚙️ Modelos Aplicados

Se exploraron y aplicaron los siguientes modelos de regresión:

Regresión Lineal
Ridge
Lasso
ElasticNet
Árbol de Decisión
Random Forest
Gradient Boosting
Modelo de Stacking (combinando RandomForest, Ridge, Lasso y Gradient Boosting) ✅ Mejor Modelo


Tras la evaluación, el modelo con el mejor desempeño fue Modelo de Stacking con un R2 de 0.63 . Inicialmente, el modelo Lasso/Lineal obtuvo un R2 de ~0.61, pero los modelos avanzados han mostrado potencial para mejorar esta métrica.

🚀 Streamlit Application

Se ha desarrollado una aplicación web interactiva utilizando Streamlit para permitir a los usuarios ingresar las características de una vivienda y obtener una predicción de su precio en tiempo real. La aplicación utiliza el mejor modelo entrenado y los objetos de preprocesamiento guardados (scaler y label encoders).

📦 Cómo Usar

Para ejecutar el Jupyter Notebook (para exploración y entrenamiento de modelos):

Asegúrate de tener instaladas las bibliotecas necesarias (pandas, scikit-learn, matplotlib, seaborn). Puedes instalarlas con: ```bash pip install pandas scikit-learn matplotlib seaborn ```
Ejecuta el Jupyter Notebook (nombre_de_tu_notebook.ipynb) en tu entorno de Jupyter.
Para ejecutar la aplicación de Streamlit:

Asegúrate de tener Streamlit instalado. Puedes instalarlo con: ```bash pip install streamlit ```
Guarda los archivos del modelo (best_model.pkl), el scaler (scaler.pkl), los label encoders (label_encoders.pkl) y la lista de columnas de entrenamiento (train_columns.pkl) en la misma carpeta que tu archivo de la aplicación Streamlit (app.py).
Abre tu terminal o Git Bash, navega hasta la carpeta donde guardaste app.py y ejecuta el siguiente comando: ```bash streamlit run app.py ```
Streamlit abrirá automáticamente una nueva pestaña en tu navegador con la aplicación en funcionamiento.

