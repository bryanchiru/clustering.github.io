# clustering.github.io
🧠 Análisis de Estado Mental con Clustering

Este proyecto implementa un modelo de clustering aplicado al análisis de salud mental, con el propósito de segmentar pacientes según su nivel de riesgo psicológico y hábitos de vida.

El laboratorio integra:

✔ Preprocesamiento completo
✔ Entrenamiento de K-Means
✔ Evaluación con inercia y silueta
✔ Interpretación clínica
✔ API funcional desplegada
✔ Integración web

📊 1. Preparación del Conjunto de Datos

El dataset sintético diseñado para este laboratorio incluye 11 variables relacionadas con salud mental y estilo de vida:

Ansiedad (0–10)

Depresión (0–10)

Estrés (0–10)

Horas de sueño

Actividad física semanal

Autoestima (0–10)

Soporte social (0–10)

Consumo de alcohol (0–7)

Ideación suicida (0–10)

Rendimiento académico (0–100)

Edad

Las variables fueron estandarizadas mediante StandardScaler para mejorar el desempeño del modelo basado en distancias.

📈 2. Selección del Número de Clusters (k)

Se entrenaron modelos K-Means para valores de 
𝑘
k entre 2 y 10.
Para cada uno se calculó:

Inercia (SSE)

Coeficiente de Silueta

Se generaron dos gráficas:

Gráfica del Codo (Inercia vs k)

Silhouette Score (Silueta vs k)

El número óptimo de clusters se seleccionó en base al mejor compromiso entre variación explicada y cohesión/separación.

🧬 3. Entrenamiento del Modelo Final

Con el valor óptimo de 
𝑘
k, se entrenó el modelo final kmeans_final, asignando un cluster a cada registro.
Se generaron estadísticas descriptivas:

Media

Desviación estándar

Valores máximos y mínimos

Tamaño de cada cluster

Estas estadísticas permitieron analizar adecuadamente cada grupo.

🧾 4. Interpretación Clínica de los Clusters
🟢 Cluster 0 – Bajo Riesgo

Baja ansiedad, depresión y estrés

Buen sueño y actividad física

Autoestima y soporte social altos

Ideación suicida mínima

➡ Perfil emocionalmente estable.

🟡 Cluster 1 – Estrés Académico / Riesgo Moderado

Estrés y ansiedad elevados

Sueño reducido

Autoestima media

Ideación suicida leve

➡ Perfil asociado a sobrecarga académica o personal.

🔴 Cluster 2 – Alto Riesgo Psicológico

Alta ansiedad, depresión y estrés

Bajo soporte social

Autoestima reducida

Ideación suicida elevada

Rendimiento afectado

➡ Necesita atención psicológica prioritaria.

🚀 5. API del Modelo (FastAPI + Render)

El modelo fue serializado como:

modelo_clustering.pkl

scaler.pkl

Y desplegado mediante FastAPI en Render:

📡 Endpoint
POST https://api-clustering.onrender.com/predecir

🔌 Ejemplo de solicitud
{
  "valores": [5, 6, 7, 6, 100, 4, 5, 2, 3, 70, 22]
}

🔍 Ejemplo de respuesta
{
  "cluster": 0,
  "variables": [...],
  "valores": [...]
}

🌐 6. Integración Web

La API se conecta con una interfaz web que permite ingresar valores manualmente.
El sistema devuelve el cluster asignado en tiempo real, convirtiéndolo en una herramienta interactiva útil para análisis académico y demostración.

📁 7. Estructura del Repositorio
clustering.github.io/
│
├── modelo/
│   ├── modelo_clustering.pkl
│   └── scaler.pkl
│
├── main.py
├── requirements.txt
├── Analisisdeestadomental.ipynb
└── README.md

🧩 8. Limitaciones

Dataset sintético (no datos reales)

K-Means no captura estructuras complejas

No se incluye información cualitativa

No sustituye evaluación clínica profesional

🧠 9. Conclusiones

Este laboratorio permitió aplicar técnicas de clustering para la segmentación de pacientes según sus indicadores de salud mental.

El proceso evidenció:

La importancia del preprocesamiento

La utilidad del clustering para identificar perfiles psicológicos

La viabilidad de integrar modelos con APIs

Cómo una solución completa puede abarcar:
modelo → servidor → interfaz web

El resultado es un sistema funcional, interactivo y útil como demostración académica.

👤 Autor

Bryan Chirú V
Ingeniería Biomédica – ULAT
Inteligencia Artificial – 2025
