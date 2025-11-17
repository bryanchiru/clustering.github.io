# clustering.github.io
🧠 Análisis de Estado Mental con Clustering 

Este proyecto implementa un modelo de clustering aplicado al análisis de salud mental, con el propósito de segmentar a los pacientes según su nivel de riesgo psicológico, comportamiento emocional y hábitos de vida.

El laboratorio integra:

✔ Preprocesamiento completo del dataset
✔ Entrenamiento de K-Means
✔ Evaluación con inercia y coeficiente de silueta
✔ Interpretación clínica de cada grupo
✔ Despliegue del modelo usando FastAPI
✔ API funcional publicada en Render
✔ Conexión con una interfaz web

📊 1. Preparación del Conjunto de Datos

El dataset utilizado es sintético, diseñado para representar pacientes con distintos perfiles psicológicos y conductuales. Incluye 11 variables:

Ansiedad (0–10)

Depresión (0–10)

Estrés (0–10)

Horas de sueño por noche

Minutos de actividad física por semana

Autoestima (0–10)

Soporte social (0–10)

Consumo de alcohol (0–7 días/semana)

Ideación suicida (0–10)

Rendimiento académico (0–100)

Edad

Antes del modelado, se aplicó StandardScaler para estandarizar las variables y facilitar el aprendizaje basado en distancias.

📈 2. Selección del Número Óptimo de Clusters

Se evaluaron modelos K-Means para valores de 
𝑘
k entre 2 y 10. Para cada uno se calcularon:

Inercia (SSE)

Coeficiente de Silueta

Con estos datos se generaron dos gráficas:

Método del codo (Inercia vs k)

Curva de Silueta (Silhouette vs k)

El número óptimo de clusters se seleccionó con base en:

📌 Buen equilibrio entre baja inercia y alta cohesión/separación.

🧬 3. Entrenamiento del Modelo Final

Ya con el valor óptimo de 
𝑘
k, se entrenó el modelo definitivo kmeans_final.
Cada paciente fue clasificado y se generaron estadísticas clave:

Media por variable

Desviación estándar

Valores extremos

Cantidad de pacientes por cluster

Este análisis permitió interpretar el significado clínico de cada grupo.

🧾 4. Interpretación Clínica de los Clusters

Los clusters obtenidos muestran una separación clara entre perfiles de riesgo bajo, moderado y alto.

🟢 Cluster 0 – Bajo riesgo / Perfil compensado

Estrés, ansiedad y depresión bajos

Buenos hábitos de sueño y actividad física

Alta autoestima y soporte social

Ideación suicida mínima

Rendimiento académico alto

➡ Representa pacientes bien regulados emocionalmente.

🟡 Cluster 1 – Estrés académico / Riesgo moderado

Ansiedad y estrés elevados

Sueño reducido

Autoestima y soporte medio

Ideación suicida leve

Rendimiento académico variable

➡ Grupo en riesgo medio, caracterizado por sobrecarga académica.

🔴 Cluster 2 – Alto riesgo psicológico

Altos niveles de ansiedad, depresión y estrés

Autoestima baja

Poca actividad física

Soporte social limitado

Ideación suicida alta

Bajo rendimiento académico

➡ Representa pacientes con clara necesidad de intervención prioritaria.

🚀 5. API del Modelo – FastAPI + Render

El modelo y el escalador fueron serializados como:

modelo_clustering.pkl

scaler.pkl

Y se implementó una API con FastAPI, desplegada en Render:

📡 Endpoint principal
POST https://api-clustering.onrender.com/predecir

🔌 Ejemplo de petición
{
  "valores": [5, 6, 7, 6, 100, 4, 5, 2, 3, 70, 22]
}

🔍 Ejemplo de respuesta
{
  "cluster": 0,
  "variables": [...],
  "valores": [...]
}


La API se integra exitosamente con una interfaz web funcional.

🌐 6. Integración con Interfaz Web

La página web del proyecto incluye una sección donde el usuario ingresa las 11 variables y el sistema envía la petición a la API.
El resultado muestra el cluster asignado, permitiendo una consulta simple, rápida y visual.

📁 7. Estructura del Repositorio
clustering.github.io/
│
├── modelo/
│    ├── modelo_clustering.pkl
│    └── scaler.pkl
│
├── main.py
├── requirements.txt
├── Analisisdeestadomental.ipynb
└── README.md

🧩 8. Limitaciones del Modelo

Aunque funcional y útil como ejercicio académico, el sistema presenta algunas limitaciones:

El dataset es sintético, no clínico real.

No se incorpora información cualitativa o historial previo.

El modelo no reemplaza evaluación profesional.

K-Means supone clusters esféricos y puede fallar si los datos reales no cumplen esa forma.

🧠 9. Conclusiones

El proyecto permitió aplicar de manera práctica técnicas de clustering para segmentar pacientes según indicadores psicológicos.
La metodología implementada demostró:

El valor del preprocesamiento (escalado, normalización)

La importancia de seleccionar correctamente el número de clusters

La capacidad del clustering para revelar patrones de riesgo

Que la integración con APIs puede convertir modelos analíticos en herramientas interactivas

El potencial del análisis automatizado como apoyo en la toma de decisiones clínicas

El despliegue web y la API completan la solución, permitiendo un flujo completo: modelo → servidor → interfaz web.

👤 Autor

Bryan Chirú V
Ingeniería Biomédica – ULAT
Inteligencia Artificial, 2025
