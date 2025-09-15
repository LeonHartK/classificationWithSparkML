# Modularized version using separated concerns

from config.spark_session import get_spark_session
from data.data_loader import load_data, show_basic_info
from evaluation.evaluation import evaluate
from features.preprocessing import preprocess_data
from models.model import train_model, predict


def main():
    spark = get_spark_session()
    file_path = "Data/adult_income_sample.csv"
    df = load_data(spark, file_path)
    show_basic_info(df)

    categorical_cols = ["sex", "workclass", "education"]
    numerical_cols = ["age", "fnlwgt", "hours_per_week"]
    indexers, encoders, assembler, feature_cols = preprocess_data(
        df, categorical_cols, numerical_cols
    )

    print(f"\nColumnas numéricas incluidas: {numerical_cols}")
    print(
        f"Columnas categóricas codificadas: {[col + '_encoded' for col in categorical_cols]}"
    )

    model = train_model(df, indexers, encoders, assembler)
    predictions = predict(model, df)
    evaluate(predictions)


if __name__ == "__main__":
    main()
