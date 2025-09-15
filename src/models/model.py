from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.sql import DataFrame


def train_model(df: DataFrame, indexers, encoders, assembler):
    lr = LogisticRegression(
        featuresCol="features",
        labelCol="label_indexed",
        predictionCol="prediction",
        probabilityCol="probability",
        rawPredictionCol="rawPrediction",
        maxIter=100,
        regParam=0.01,
        elasticNetParam=0.0,
    )
    pipeline = Pipeline(stages=indexers + encoders + [assembler, lr])
    model = pipeline.fit(df)
    return model


def predict(model, df: DataFrame):
    predictions = model.transform(df)
    return predictions
