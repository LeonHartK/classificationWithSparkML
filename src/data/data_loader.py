from pyspark.sql import DataFrame
from pyspark.sql import SparkSession


def load_data(spark: SparkSession, file_path: str) -> DataFrame:
    df = spark.read.csv(file_path, header=True, inferSchema=True)
    return df


def split_data(df: DataFrame, seed=42):
    # Split: 10% test, 90% rest; then 20% train, 80% validation from the rest
    splits = df.randomSplit([0.1, 0.9], seed=seed)
    test_df = splits[0]
    rest_df = splits[1]
    train_val_splits = rest_df.randomSplit([0.2, 0.8], seed=seed)
    train_df = train_val_splits[0]
    val_df = train_val_splits[1]
    return train_df, val_df, test_df


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
