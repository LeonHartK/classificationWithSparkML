from pyspark.sql import DataFrame


def evaluate(predictions: DataFrame):
    predictions.select(
        "age",
        "sex",
        "education",
        "hours_per_week",
        "label",
        "prediction",
        "probability",
    ).show(10, truncate=False)
    print("\nMétricas de evaluación:")
    correct_predictions = predictions.filter(
        predictions.label_indexed == predictions.prediction
    ).count()
    total = predictions.count()
    accuracy = correct_predictions / total
    print(f" Exactitud del modelo: {accuracy:.4f}")
    predictions.groupBy("label", "prediction").count().orderBy(
        "label", "prediction"
    ).show()
