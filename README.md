Proyecto: Limpieza de Datos y Predicción de Tiempos de Maratón
Resumen:
Este proyecto se enfoca en la limpieza y análisis de un conjunto de datos de maratones. Se implementaron modelos de regresión para predecir los tiempos de finalización. El conjunto de datos se limpió, transformó y se evaluaron modelos de Regresión Lineal, Ridge y Lasso.
Pasos Destacados:
Limpieza de Datos:
Se manejaron valores faltantes en 'Category' y 'CrossTraining'.
Se transformaron datos, convirtiendo 'CrossTraining' a valores numéricos.
Se filtraron datos con 'sp4week' < 1000.
Modelado de Regresión:
Se dividió el conjunto de datos y se evaluaron modelos de regresión.
El modelo Ridge fue seleccionado por su buen rendimiento.
Persistencia del Modelo:

Se guardó el modelo Ridge con pickle para futuras predicciones.
Resultados:
Modelo Ridge:
R-cuadrado en entrenamiento: 0.9527, en prueba: 0.9375.
Archivos en el Repositorio:
Cuaderno Jupyter:

notebooks/Modelo.ipynb

Modelo Guardado:

produccion/ridge_model.pkl

Conjunto de Datos:

MarathonData.csv

Instrucciones:
Ejecute el cuaderno Jupyter para explorar el análisis y los modelos.
Utilice el modelo guardado en predict.py para realizar predicciones.

Autor:
Christian Villacis

Contacto:
cvillacisvanegas@gmail.com
