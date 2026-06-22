# Actividad5

## Descripción
Proyecto de entrenamiento y evaluación de modelos de ML con seguimiento de experimentos mediante MLflow.

## Estructura
```
Actividad5/
├── datos/
│   ├── datos_ini/      # Datos originales sin procesar
│   └── datos_limp/     # Datos limpios listos para entrenamiento
├── fuentes/
│   ├── entrena.ipynb   # Notebook principal (Colab)
│   ├── datos_prep.py   # Funciones de limpieza y preprocesamiento
│   └── train.py        # Script de entrenamiento y registro en MLflow
├── README.md
└── CHANGELOG.md
```

## Requisitos
```
pandas
scikit-learn
mlflow
jupyter
```

Instalar con:
```bash
pip install pandas scikit-learn mlflow jupyter
```

## Reproducción paso a paso

1. **Colocar datos originales** en `datos/datos_ini/`.

2. **Limpiar datos:**
   ```python
   from fuentes.datos_prep import pipeline_limpieza
   pipeline_limpieza('datos/datos_ini/datos.csv', 'datos/datos_limp/datos.csv')
   ```

3. **Entrenar y registrar en MLflow:**
   ```bash
   python fuentes/train.py
   ```

4. **Ver experimentos en MLflow UI:**
   ```bash
   mlflow ui
   ```
   Abrir http://localhost:5000 en el navegador.

5. **Notebook en Colab:** Abrir `fuentes/entrena.ipynb` y ejecutar las celdas en orden.

## Decisiones de diseño
- La limpieza de datos está separada del entrenamiento para facilitar la reproducibilidad.
- Se usa MLflow para trackear hiperparámetros, métricas y artefactos de cada experimento.
- El notebook de Colab importa las funciones de `datos_prep.py` y `train.py` para evitar duplicar lógica.
