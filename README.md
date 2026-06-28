# Actividad 5 — Preparación de datos, entrenamiento y registro en MLflow

Proyecto aplicado de gestión de datos y MLOps: se selecciona un dataset público,
se prepara y versiona, se entrenan **dos modelos**, se ajustan hiperparámetros y se
registran los experimentos en **MLflow**, garantizando trazabilidad y reproducibilidad.

- **Dataset:** Titanic (público, vía `seaborn.load_dataset('titanic')`).
- **Problema:** Clasificación binaria (`survived`).
- **Modelos comparados:** Regresión Logística vs Random Forest.
- **Métricas:** accuracy, precision, recall, F1, ROC-AUC.
- **Validación:** GridSearchCV + validación cruzada estratificada (5 folds).
- **Comparación estadística:** prueba t pareada y prueba de Wilcoxon.

## Estructura del repositorio

```
Actividad5/
├── datos/
│   ├── datos_ini/titanic_raw.csv     # datos originales (crudos)
│   └── datos_limp/titanic_clean.csv  # datos limpios y estandarizados
├── fuentes/
│   ├── entrena.ipynb                 # notebook Colab (flujo completo)
│   ├── datos_prep.py                 # funciones de limpieza
│   └── train.py                      # entrenamiento + registro en MLflow
├── evidencias/                       # figuras y tablas comparativas
├── README.md
└── CHANGELOG.md
```

## Descripción del dataset (diccionario)

| Columna | Descripción | Tipo |
|---|---|---|
| survived | Sobrevivió (1) o no (0) — **variable objetivo** | int |
| pclass | Clase del boleto (1, 2, 3) | int |
| sex | Sexo (0=hombre, 1=mujer tras codificar) | int |
| age | Edad en años (imputada con la mediana) | float |
| sibsp | Hermanos/cónyuge a bordo | int |
| parch | Padres/hijos a bordo | int |
| fare | Tarifa pagada | float |
| family_size | `sibsp + parch + 1` (variable derivada) | int |
| emb_C / emb_Q / emb_S | Puerto de embarque (one-hot) | int |

**Supuestos:** se asume que `alive`, `class`, `embark_town`, `adult_male`, `who` y
`alone` son redundantes/derivadas y que `deck` (~77% nulos) no aporta señal confiable.

## Proceso de limpieza y estandarización (`datos_prep.py`)

1. **Eliminación de columnas** redundantes o con fuga de información (`alive` es copia de `survived`).
2. **Reglas de negocio:** se descartan `fare < 0` y `age <= 0`.
3. **Valores nulos:** `age` y `fare` → mediana; `embarked` → moda.
4. **Ingeniería de variables:** `family_size`.
5. **Codificación categórica:** `sex` → 0/1; `embarked` → one-hot.
6. **Escalado:** `StandardScaler` se aplica **dentro del Pipeline** de entrenamiento
   (solo afecta a la Regresión Logística) para evitar fuga del conjunto de prueba.

Resultado: 891 registros, 11 columnas, **0 nulos**.

## Reproducción

Requisitos: Python 3.x, `pip install mlflow scikit-learn pandas seaborn scipy matplotlib`.

```bash
# 1) Servidor MLflow (en una terminal aparte)
python -m mlflow server --host 127.0.0.1 --port 5000 \
  --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns

# 2) Generar datasets (crudo + limpio)
python fuentes/datos_prep.py

# 3) Entrenar, ajustar y registrar en MLflow
python fuentes/train.py
```

Abre el panel en **http://127.0.0.1:5000** (experimento `Actividad5_Titanic`).
La semilla `random_state=42` garantiza resultados reproducibles.

## Resultados (conjunto de prueba)

| Modelo | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Regresión Logística | 0.804 | 0.793 | 0.667 | 0.724 | 0.844 |
| Random Forest | 0.804 | 0.783 | 0.681 | 0.729 | 0.839 |

**Comparación estadística (F1 sobre 5 folds):** prueba t pareada p≈0.07 y Wilcoxon
p≈0.13 → la diferencia **no es estadísticamente significativa** (α=0.05). Ambos
modelos son equivalentes; se prefiere la Regresión Logística por simplicidad e
interpretabilidad (menor deuda técnica).

Ver `evidencias/` para figuras (EDA, matrices de confusión, curva ROC) y capturas de MLflow.
