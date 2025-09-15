from pyspark.sql import SparkSession


def get_spark_session(
    app_name: str = "ClasificacionIngresos", shuffle_partitions: int = 100
) -> SparkSession:
    spark = (
        SparkSession.builder.appName(app_name)
        .config("spark.sql.shuffle.partitions", str(shuffle_partitions))
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark
