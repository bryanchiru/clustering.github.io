from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
from fastapi.middleware.cors import CORSMiddleware  # ⬅ NUEVO

FEATURES = [
    "ansiedad",
    "depresion",
    "estres",
    "horas_sueno",
    "actividad_fisica",
    "autoestima",
    "soporte_social",
    "consumo_alcohol",
    "ideacion_suicida",
    "rendimiento_academico",
    "edad"
]

app = FastAPI()

# 🔓 CORS para permitir llamadas desde tu página
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # si quieres, luego lo cambiamos a tu URL específica
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

modelo = joblib.load("modelo/modelo_clustering.pkl")
scaler = joblib.load("modelo/scaler.pkl")

class DatosPaciente(BaseModel):
    valores: list  # lista de 11 valores en el orden de FEATURES

@app.get("/")
def root():
    return {"mensaje": "API de clustering lista y funcionando"}

@app.post("/predecir")
def predecir(datos: DatosPaciente):
    x = np.array(datos.valores).reshape(1, -1)
    x_scaled = scaler.transform(x)
    cluster = int(modelo.predict(x_scaled)[0])
    return {
        "cluster": cluster,
        "variables": FEATURES,
        "valores": datos.valores
    }
