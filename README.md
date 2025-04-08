# 🏠 House Price Prediction - Machine Learning

Este proyecto predice el precio de casas basado en sus características (metros cuadrados, baños, habitaciones, etc.) utilizando varios modelos de regresión. Fue realizado como parte de mi portfolio de ciencia de datos.

## 📌 Dataset

Contiene columnas como:
- `price`: precio en moneda local
- `area`: superficie en metros cuadrados
- `bedrooms`, `bathrooms`, `stories`, `parking`: características numéricas
- `guestroom`, `hotwaterheating`, `airconditioning`: booleanas

## ⚙️ Modelos aplicados

- Regresión Lineal
- Ridge
- Lasso
- ElasticNet
- Árbol de Decisión
- Random Forest

## ✅ Mejor modelo

El mejor desempeño lo obtuvo el modelo **Lasso/Lineal**, con un R2 de ~0.61.

## 📦 Cómo usar

```bash
# Instalar dependencias
pip install -r requirements.txt

# Cargar y usar el modelo entrenado
from joblib import load
modelo = load("mejor_modelo.pkl")
scaler = load("scaler.pkl")
X_escalado = scaler.transform(X_nuevo)
predicciones = modelo.predict(X_escalado)
