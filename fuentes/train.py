import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score


RUTA_DATOS = "../datos/datos_limp/datos.csv"
EXPERIMENTO = "Actividad5"
TARGET = "target"


def cargar_datos(ruta: str, target: str):
    df = pd.read_csv(ruta)
    X = df.drop(columns=[target])
    y = df[target]
    return train_test_split(X, y, test_size=0.2, random_state=42)


def entrenar(modelo, X_train, y_train):
    modelo.fit(X_train, y_train)
    return modelo


def evaluar(modelo, X_test, y_test) -> dict:
    y_pred = modelo.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred, average="weighted"),
    }


def main():
    mlflow.set_experiment(EXPERIMENTO)

    X_train, X_test, y_train, y_test = cargar_datos(RUTA_DATOS, TARGET)

    # Reemplazar con el modelo deseado
    from sklearn.ensemble import RandomForestClassifier
    modelo = RandomForestClassifier(n_estimators=100, random_state=42)

    with mlflow.start_run():
        mlflow.log_params(modelo.get_params())

        modelo = entrenar(modelo, X_train, y_train)
        metricas = evaluar(modelo, X_test, y_test)

        mlflow.log_metrics(metricas)
        mlflow.sklearn.log_model(modelo, "modelo")

        print("Métricas:", metricas)


if __name__ == "__main__":
    main()
