from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

# Cargar modelo
modelo = joblib.load("modelo_clustering.pkl")

# Datos que enviará la web
class DatosUsuario(BaseModel):
    Sleep_Duration: float
    Quality_of_Sleep: float
    Stress_Level: float
    Physical_Activity_Level: float
    Heart_Rate: float
    Daily_Steps: float

@app.post("/predict")
def predict(data: DatosUsuario):
    valores = np.array([[ 
        data.Sleep_Duration,
        data.Quality_of_Sleep,
        data.Stress_Level,
        data.Physical_Activity_Level,
        data.Heart_Rate,
        data.Daily_Steps
    ]])

    cluster = modelo.predict(valores)[0]
    
    return {"cluster": int(cluster)}
