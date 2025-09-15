from pyspark.sql import DataFrame
from pyspark.sql import SparkSession


def load_data(spark: SparkSession, file_path: str) -> DataFrame:
    df = spark.read.csv(file_path, header=True, inferSchema=True)
    return df


def show_basic_info(df: DataFrame):
    print(f"Dataset cargado con {df.count()} registros")
    print("Esquema del DataFrame:")
    df.printSchema()
    print("Primeros 5 registros del DataFrame:")
    df.show(5, truncate=False)
    print("Estadísticas descriptivas de las columnas numéricas:")
    df.select("age", "fnlwgt", "hours_per_week").describe().show()
    print("Conteo de registros por etiqueta (label):")
    df.groupBy("label").count().show()
