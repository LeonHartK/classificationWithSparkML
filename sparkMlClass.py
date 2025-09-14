from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.sql.types import StructType, StructField, IntegerType, StringType
import pyspark.sql.functions as F

# Initialize Spark session

spark = SparkSession.builder \
    .appName("ClasificacionIngresos") \
    .config("spark.sql.shuffle.partitions", "100") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

# 1. Define schema for the dataset

file_path = "Data/adult_income_sample.csv"
df = spark.read.csv(file_path, header=True, inferSchema=True)

print(f"Dataset cargado con {df.count()} registros")

print("Esquema del DataFrame:")
df.printSchema()

print("Primeros 5 registros del DataFrame:")
df.show(5, truncate=False)

print("Estadísticas descriptivas de las columnas numéricas:")
df.select("age", "fnlwgt", "hours_per_week").describe().show()

print("Conteo de registros por etiqueta (label):")
df.groupBy("label").count().show()

# 2. Data Preprocessing

categorical_cols = ["sex", "workclass", "education"]

# StringIndexer para variables categóricas
indexers = []
for col in categorical_cols:
    indexer = StringIndexer(inputCol=col, outputCol=col + "_indexed")
    indexers.append(indexer)
    print(f" StringIndexer creado para: {col}")

# Se Indexa la variable objetivo
label_indexer = StringIndexer(inputCol="label", outputCol="label_indexed")
indexers.append(label_indexer)
print(f" StringIndexer creado para: label (variable objetivo)")

# One-Hot Encoding para variables categóricas
encoders = []
for col in categorical_cols:
    encoder = OneHotEncoder(inputCol=col + "_indexed", outputCol=col + "_encoded")
    encoders.append(encoder)
    print(f" OneHotEncoder creado para: {col}_indexed")

# 3. Feature Assembly

# Columnas numéricas
numerical_cols = ["age", "fnlwgt", "hours_per_week"]

# Columnas codificadas
encoded_cols = [col + "_encoded" for col in categorical_cols]

# Ensamblar todas las características en un solo vector
feature_cols = numerical_cols + encoded_cols
assembler = VectorAssembler(
    inputCols=feature_cols,
    outputCol="features"
)

print(f"\nColumnas numéricas incluidas: {numerical_cols}")
print(f"Columnas categóricas codificadas: {encoded_cols}")

# 4. Modelo de Clasificación

lr = LogisticRegression(
    featuresCol="features",
    labelCol="label_indexed",
    predictionCol="prediction",
    probabilityCol="probability",
    rawPredictionCol="rawPrediction",
    maxIter=100,
    regParam=0.01,
    elasticNetParam=0.0
)

# pipeline
pipeline = Pipeline(stages=indexers + encoders + [assembler, lr])

model = pipeline.fit(df)

predictions = model.transform(df)

predictions.select(
    "age", "sex", "education", "hours_per_week", 
    "label", "prediction", "probability"
).show(10, truncate=False)

print("\nMétricas de evaluación:")

correct_predictions = predictions.filter(
    predictions.label_indexed == predictions.prediction
).count()
total = predictions.count()
accuracy = correct_predictions / total

print(f" Exactitud del modelo: {accuracy:.4f}")

predictions.groupBy("label", "prediction").count().orderBy("label", "prediction").show()