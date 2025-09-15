from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.sql import DataFrame


def preprocess_data(df: DataFrame, categorical_cols=None, numerical_cols=None):
    if categorical_cols is None:
        categorical_cols = ["sex", "workclass", "education"]
    if numerical_cols is None:
        numerical_cols = ["age", "fnlwgt", "hours_per_week"]

    # StringIndexer para variables categóricas
    indexers = [
        StringIndexer(inputCol=col, outputCol=col + "_indexed")
        for col in categorical_cols
    ]
    # Indexar la variable objetivo
    label_indexer = StringIndexer(inputCol="label", outputCol="label_indexed")
    indexers.append(label_indexer)

    # One-Hot Encoding para variables categóricas
    encoders = [
        OneHotEncoder(inputCol=col + "_indexed", outputCol=col + "_encoded")
        for col in categorical_cols
    ]

    encoded_cols = [col + "_encoded" for col in categorical_cols]
    feature_cols = numerical_cols + encoded_cols
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")

    return indexers, encoders, assembler, feature_cols
