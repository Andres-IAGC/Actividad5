"""
datos_prep.py
Funciones de limpieza y estandarizacion del dataset Titanic.

Reglas aplicadas:
- Eliminacion de columnas redundantes o con fuga de informacion (leakage).
- Imputacion de valores nulos (age, fare -> mediana; embarked -> moda).
- Regla de negocio: descartar registros con valores imposibles (fare < 0, age <= 0).
- Ingenieria de variable: family_size = sibsp + parch + 1.
- Codificacion de variables categoricas (sex, embarked).

El escalado/normalizacion NO se aplica aqui: se realiza dentro del Pipeline
de entrenamiento (StandardScaler) para evitar fuga de datos del set de prueba.
"""

import pandas as pd

# Columnas que se eliminan por ser redundantes o provocar fuga de informacion.
# - alive: copia exacta de la etiqueta 'survived' (leakage directo).
# - adult_male, who: derivadas de sex/age.
# - class: copia textual de pclass.
# - embark_town: copia textual de embarked.
# - deck: 77% de valores nulos.
# - alone: derivada de sibsp/parch.
COLS_ELIMINAR = ["alive", "adult_male", "who", "class", "embark_town", "deck", "alone"]


def cargar_datos_crudos(ruta_csv):
    """Lee el CSV original tal cual fue obtenido de la fuente."""
    return pd.read_csv(ruta_csv)


def limpiar(df):
    """Aplica todo el proceso de limpieza y devuelve un DataFrame listo para modelar."""
    df = df.copy()

    # 1. Eliminar columnas redundantes / con fuga de informacion.
    df = df.drop(columns=[c for c in COLS_ELIMINAR if c in df.columns])

    # 2. Reglas de negocio: eliminar registros con valores imposibles.
    if "fare" in df.columns:
        df = df[df["fare"].isna() | (df["fare"] >= 0)]
    if "age" in df.columns:
        df = df[df["age"].isna() | (df["age"] > 0)]

    # 3. Imputacion de valores nulos.
    df["age"] = df["age"].fillna(df["age"].median())
    df["fare"] = df["fare"].fillna(df["fare"].median())
    df["embarked"] = df["embarked"].fillna(df["embarked"].mode()[0])

    # 4. Ingenieria de variables.
    df["family_size"] = df["sibsp"] + df["parch"] + 1

    # 5. Codificacion de variables categoricas.
    df["sex"] = df["sex"].map({"male": 0, "female": 1}).astype(int)
    df = pd.get_dummies(df, columns=["embarked"], prefix="emb")
    # get_dummies genera columnas bool -> convertir a int para CSV/modelo.
    bool_cols = df.select_dtypes(include="bool").columns
    df[bool_cols] = df[bool_cols].astype(int)

    # 6. Tipos finales coherentes.
    df["age"] = df["age"].astype(float)
    df["fare"] = df["fare"].astype(float)

    return df.reset_index(drop=True)


if __name__ == "__main__":
    # Ejecucion directa: genera datos_ini (crudo) y datos_limp (limpio).
    import os
    import seaborn as sns

    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ruta_ini = os.path.join(base, "datos", "datos_ini", "titanic_raw.csv")
    ruta_limp = os.path.join(base, "datos", "datos_limp", "titanic_clean.csv")

    # Obtencion del dataset publico (Titanic, via seaborn).
    crudo = sns.load_dataset("titanic")
    crudo.to_csv(ruta_ini, index=False)
    print(f"[OK] Datos crudos guardados: {ruta_ini}  {crudo.shape}")

    limpio = limpiar(crudo)
    limpio.to_csv(ruta_limp, index=False)
    print(f"[OK] Datos limpios guardados: {ruta_limp}  {limpio.shape}")
    print("Nulos restantes:", int(limpio.isna().sum().sum()))
    print("Columnas:", list(limpio.columns))
