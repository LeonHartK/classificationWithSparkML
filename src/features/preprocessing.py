from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.sql import DataFrame


def preprocess_data(df: DataFrame, categorical_cols=None, numerical_cols=None):
    if categorical_cols is None:
        categorical_cols = ["sex", "workclass", "education"]
    if numerical_cols is None:
        numerical_cols = ["age", "fnlwgt", "hours_per_week"]

    # Feature engineering: interaction and polynomial features
    from pyspark.sql.functions import col

    # Use StringIndexer to convert categorical columns to numeric before interaction
    from pyspark.ml.feature import StringIndexer

    # Only index for interaction if not already present
    if "workclass_indexed" not in df.columns:
        workclass_indexer = StringIndexer(
            inputCol="workclass", outputCol="workclass_indexed"
        )
        df = workclass_indexer.fit(df).transform(df)
    if "education_indexed" not in df.columns:
        education_indexer = StringIndexer(
            inputCol="education", outputCol="education_indexed"
        )
        df = education_indexer.fit(df).transform(df)
    df = df.withColumn(
        "workclass_education_interaction",
        col("workclass_indexed") * col("education_indexed"),
    )
    df = df.withColumn(
        "hours_per_week_squared", col("hours_per_week") * col("hours_per_week")
    )
    numerical_cols += ["workclass_education_interaction", "hours_per_week_squared"]

    # StringIndexer for categorical variables (skip workclass and education, already indexed)
    indexers = [
        StringIndexer(inputCol=col, outputCol=col + "_indexed")
        for col in categorical_cols
        if col not in ["workclass", "education"]
    ]
    label_indexer = StringIndexer(inputCol="label", outputCol="label_indexed")
    indexers.append(label_indexer)

    # One-Hot Encoding for categorical variables
    encoders = [
        OneHotEncoder(inputCol=col + "_indexed", outputCol=col + "_encoded")
        for col in categorical_cols
    ]

    encoded_cols = [col + "_encoded" for col in categorical_cols]
    feature_cols = numerical_cols + encoded_cols
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")

    return df, indexers, encoders, assembler, feature_cols
