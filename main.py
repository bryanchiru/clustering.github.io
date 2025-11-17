from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np

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

# DESCRIPCIONES DE CADA CLUSTER (ejemplo académico, NO diagnóstico real)
cluster_labels = {
    0: {
        "nombre": "Bajo riesgo / perfil compensado",
        "descripcion": (
            "Perfil con niveles bajos de ansiedad, depresión y estrés, buenas horas de sueño "
            "y actividad física adecuada. Autoestima y soporte social suelen ser buenos, con "
            "ideación suicida muy baja."
        ),
        "riesgo_suicidio": 5,
        "riesgo_mortalidad": 2,
        "recomendacion": (
            "Mantener hábitos saludables, monitoreo preventivo y educación en salud mental. "
            "No se identifica un riesgo elevado inmediato."
        ),
    },
    1: {
        "nombre": "Estrés académico / riesgo moderado",
        "descripcion": (
            "Perfil con niveles moderados-altos de estrés y ansiedad, sueño reducido y posible "
            "impacto en el rendimiento académico. Puede existir ideación suicida leve y sensación "
            "de sobrecarga."
        ),
        "riesgo_suicidio": 35,
        "riesgo_mortalidad": 15,
        "recomendacion": (
            "Recomendable apoyo psicológico, manejo del estrés, higiene del sueño y fortalecimiento "
            "del soporte social. Riesgo moderado si no se interviene."
        ),
    },
    2: {
        "nombre": "Alto riesgo psicológico",
        "descripcion": (
            "Perfil con niveles altos de ansiedad, depresión y estrés, autoestima baja, poco "
            "soporte social e ideación suicida elevada. El rendimiento académico y funcional "
            "suele estar significativamente deteriorado."
        ),
        "riesgo_suicidio": 80,
        "riesgo_mortalidad": 45,
        "recomendacion": (
            "Requiere intervención prioritaria, evaluación clínica completa y posible derivación a "
            "servicios especializados. Riesgo elevado."
        ),
    },
}

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # si quieres luego lo restringes a tu dominio
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cargar modelo y scaler
modelo = joblib.load("modelo/modelo_clustering.pkl")
scaler = joblib.load("modelo/scaler.pkl")


class DatosPaciente(BaseModel):
    valores: list[float]


@app.get("/")
def root():
    return {"mensaje": "API de clustering de salud mental lista"}


@app.post("/predecir")
def predecir(datos: DatosPaciente):
    x = np.array(datos.valores).reshape(1, -1)
    x_scaled = scaler.transform(x)
    cluster = int(modelo.predict(x_scaled)[0])

    info = cluster_labels.get(
        cluster,
        {
            "nombre": "Cluster no definido",
            "descripcion": "No hay descripción disponible para este cluster.",
            "riesgo_suicidio": 0,
            "riesgo_mortalidad": 0,
            "recomendacion": "Sin recomendación disponible.",
        },
    )

    return {
        "cluster": cluster,
        "nombre_cluster": info["nombre"],
        "descripcion_cluster": info["descripcion"],
        "riesgo_suicidio": info["riesgo_suicidio"],
        "riesgo_mortalidad": info["riesgo_mortalidad"],
        "recomendacion": info["recomendacion"],
        "variables": FEATURES,
        "valores": datos.valores,
    }
