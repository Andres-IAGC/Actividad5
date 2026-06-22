import pandas as pd


def cargar_datos(ruta: str) -> pd.DataFrame:
    return pd.read_csv(ruta)


def eliminar_nulos(df: pd.DataFrame, umbral: float = 0.5) -> pd.DataFrame:
    limite = int(len(df.columns) * umbral)
    df = df.dropna(thresh=limite)
    df = df.dropna(axis=1, thresh=int(len(df) * umbral))
    return df


def normalizar(df: pd.DataFrame, columnas: list) -> pd.DataFrame:
    for col in columnas:
        minimo = df[col].min()
        maximo = df[col].max()
        if maximo != minimo:
            df[col] = (df[col] - minimo) / (maximo - minimo)
    return df


def pipeline_limpieza(ruta_entrada: str, ruta_salida: str, columnas_num: list = None) -> pd.DataFrame:
    df = cargar_datos(ruta_entrada)
    df = eliminar_nulos(df)
    if columnas_num:
        df = normalizar(df, columnas_num)
    df.to_csv(ruta_salida, index=False)
    return df
