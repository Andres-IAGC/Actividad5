"""
train.py
Entrenamiento, ajuste de hiperparametros (GridSearch + validacion cruzada),
evaluacion y registro de experimentos en MLflow.

Modelos comparados:
  1. Regresion Logistica (con StandardScaler).
  2. Random Forest.

Metricas (clasificacion): accuracy, precision, recall, F1, ROC-AUC.
Comparacion estadistica: prueba t pareada y prueba de Wilcoxon sobre los
scores de validacion cruzada (5 folds).

Requiere el servidor MLflow corriendo en http://127.0.0.1:5000
"""

import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import mlflow
import mlflow.sklearn

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Configuracion
# ---------------------------------------------------------------------------
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUTA_LIMP = os.path.join(BASE, "datos", "datos_limp", "titanic_clean.csv")
DIR_EVID = os.path.join(BASE, "evidencias")
os.makedirs(DIR_EVID, exist_ok=True)

SEED = 42
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Actividad5_Titanic")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)


def cargar_xy():
    df = pd.read_csv(RUTA_LIMP)
    X = df.drop(columns=["survived"])
    y = df["survived"]
    return X, y


def evaluar(modelo, X_test, y_test):
    """Calcula las metricas estandar de clasificacion sobre el set de prueba."""
    pred = modelo.predict(X_test)
    proba = modelo.predict_proba(X_test)[:, 1]
    return {
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred),
        "recall": recall_score(y_test, pred),
        "f1": f1_score(y_test, pred),
        "roc_auc": roc_auc_score(y_test, proba),
    }


def graficar_matriz(modelo, X_test, y_test, nombre):
    fig, ax = plt.subplots(figsize=(4, 4))
    ConfusionMatrixDisplay.from_estimator(modelo, X_test, y_test, ax=ax, colorbar=False)
    ax.set_title(f"Matriz de confusion - {nombre}")
    ruta = os.path.join(DIR_EVID, f"matriz_confusion_{nombre}.png")
    fig.tight_layout()
    fig.savefig(ruta, dpi=120)
    plt.close(fig)
    return ruta


def graficar_roc(resultados, X_test, y_test):
    fig, ax = plt.subplots(figsize=(5, 5))
    for nombre, modelo in resultados.items():
        proba = modelo.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, proba)
        auc = roc_auc_score(y_test, proba)
        ax.plot(fpr, tpr, label=f"{nombre} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_xlabel("Tasa de falsos positivos")
    ax.set_ylabel("Tasa de verdaderos positivos")
    ax.set_title("Curva ROC - comparacion de modelos")
    ax.legend()
    ruta = os.path.join(DIR_EVID, "curva_roc_comparacion.png")
    fig.tight_layout()
    fig.savefig(ruta, dpi=120)
    plt.close(fig)
    return ruta


def main():
    X, y = cargar_xy()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )

    # Definicion de los modelos y su malla de hiperparametros.
    configs = {
        "RegresionLogistica": {
            "pipeline": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=1000, random_state=SEED)),
            ]),
            "grid": {
                "clf__C": [0.01, 0.1, 1, 10],
                "clf__penalty": ["l2"],
            },
        },
        "RandomForest": {
            "pipeline": Pipeline([
                ("clf", RandomForestClassifier(random_state=SEED)),
            ]),
            "grid": {
                "clf__n_estimators": [100, 300],
                "clf__max_depth": [None, 5, 10],
                "clf__min_samples_split": [2, 5],
            },
        },
    }

    modelos_entrenados = {}
    metricas_tabla = {}
    cv_scores = {}

    for nombre, cfg in configs.items():
        with mlflow.start_run(run_name=nombre):
            grid = GridSearchCV(
                cfg["pipeline"], cfg["grid"], cv=cv, scoring="f1", n_jobs=-1
            )
            grid.fit(X_train, y_train)
            mejor = grid.best_estimator_

            # Scores de validacion cruzada (para comparacion estadistica).
            scores = cross_val_score(mejor, X_train, y_train, cv=cv, scoring="f1")
            cv_scores[nombre] = scores

            metricas = evaluar(mejor, X_test, y_test)
            metricas_tabla[nombre] = metricas
            modelos_entrenados[nombre] = mejor

            # Registro en MLflow: parametros, metricas, artefactos y modelo.
            mlflow.log_params(grid.best_params_)
            mlflow.log_param("cv_folds", 5)
            mlflow.log_metric("cv_f1_mean", scores.mean())
            mlflow.log_metric("cv_f1_std", scores.std())
            for k, v in metricas.items():
                mlflow.log_metric(f"test_{k}", v)

            ruta_mc = graficar_matriz(mejor, X_test, y_test, nombre)
            mlflow.log_artifact(ruta_mc, artifact_path="figuras")
            mlflow.sklearn.log_model(mejor, name="modelo")

            print(f"\n=== {nombre} ===")
            print("Mejores hiperparametros:", grid.best_params_)
            print("Metricas test:", {k: round(v, 4) for k, v in metricas.items()})

    # Curva ROC comparativa (se registra en una corrida aparte).
    ruta_roc = graficar_roc(modelos_entrenados, X_test, y_test)

    # -------------------------------------------------------------------
    # Comparacion estadistica entre los dos modelos (sobre los 5 folds).
    # -------------------------------------------------------------------
    a, b = list(cv_scores.values())
    nombres = list(cv_scores.keys())
    t_stat, t_p = stats.ttest_rel(a, b)
    try:
        w_stat, w_p = stats.wilcoxon(a, b)
    except ValueError:
        w_stat, w_p = float("nan"), float("nan")

    with mlflow.start_run(run_name="Comparacion_estadistica"):
        mlflow.log_metric("ttest_pvalue", t_p)
        mlflow.log_metric("wilcoxon_pvalue", w_p)
        mlflow.log_artifact(ruta_roc, artifact_path="figuras")

    # Tabla comparativa de metricas -> CSV de evidencia.
    tabla = pd.DataFrame(metricas_tabla).T.round(4)
    tabla.to_csv(os.path.join(DIR_EVID, "comparacion_metricas.csv"))

    print("\n================ COMPARACION ================")
    print(tabla)
    print(f"\nF1 CV {nombres[0]}: {a.mean():.4f} +/- {a.std():.4f}")
    print(f"F1 CV {nombres[1]}: {b.mean():.4f} +/- {b.std():.4f}")
    print(f"Prueba t pareada:  t={t_stat:.4f}  p={t_p:.4f}")
    print(f"Prueba Wilcoxon:   W={w_stat:.4f}  p={w_p:.4f}")
    sig = "SI" if t_p < 0.05 else "NO"
    print(f"Diferencia estadisticamente significativa (alpha=0.05): {sig}")
    print("\nEvidencias guardadas en:", DIR_EVID)
    print("Revisa el panel MLflow en http://127.0.0.1:5000")


if __name__ == "__main__":
    main()
