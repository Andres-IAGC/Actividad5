# CHANGELOG — Actividad 5

Registro de cambios del proyecto (estrategia de versionamiento de datos y código).

## [1.0.0] — 2026-06-28

### Datos
- **datos_ini/titanic_raw.csv:** obtención del dataset público Titanic (891×15) sin modificar.
- **datos_limp/titanic_clean.csv:** versión limpia (891×11, 0 nulos):
  - Eliminadas columnas redundantes/con fuga: `alive`, `class`, `embark_town`,
    `adult_male`, `who`, `alone`, `deck`.
  - Imputación de nulos: `age`/`fare` → mediana; `embarked` → moda.
  - Regla de negocio: descarte de `fare < 0` y `age <= 0`.
  - Variable derivada `family_size`; codificación de `sex` y `embarked`.

### Código
- `fuentes/datos_prep.py`: funciones de limpieza y generación de ambos datasets.
- `fuentes/train.py`: entrenamiento de Regresión Logística y Random Forest con
  GridSearchCV (validación cruzada estratificada, 5 folds) y registro en MLflow.
- `fuentes/entrena.ipynb`: notebook equivalente para Google Colab.

### Experimentos (MLflow — experimento `Actividad5_Titanic`)
- Run `RegresionLogistica`: F1 test 0.724 | ROC-AUC 0.844.
- Run `RandomForest`: F1 test 0.729 | ROC-AUC 0.839.
- Run `Comparacion_estadistica`: prueba t pareada (p≈0.07) y Wilcoxon (p≈0.13).
- Artefactos registrados: matrices de confusión, curva ROC y modelos serializados.

### Conclusión de la versión
Ambos modelos son estadísticamente equivalentes; se documenta la Regresión
Logística como modelo recomendado por interpretabilidad y menor deuda técnica.
