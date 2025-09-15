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

    # Split data
    from data.data_loader import split_data

    train_df, val_df, test_df = split_data(df, seed=42)

    categorical_cols = ["sex", "workclass", "education"]
    numerical_cols = ["age", "fnlwgt", "hours_per_week"]

    # Apply preprocessing and feature engineering to each split
    train_df, indexers, encoders, assembler, feature_cols = preprocess_data(
        train_df, categorical_cols, numerical_cols
    )
    val_df, _, _, _, _ = preprocess_data(val_df, categorical_cols, numerical_cols)
    test_df, _, _, _, _ = preprocess_data(test_df, categorical_cols, numerical_cols)

    print(f"\nColumnas numéricas incluidas: {numerical_cols}")
    print(
        f"Columnas categóricas codificadas: {[col + '_encoded' for col in categorical_cols]}"
    )

    # Train on train_df, validate on val_df, test on test_df
    model = train_model(train_df, indexers, encoders, assembler)
    val_predictions = predict(model, val_df)
    test_predictions = predict(model, test_df)
    print("Validation Results:")
    evaluate(val_predictions)
    print("Test Results:")
    evaluate(test_predictions)


if __name__ == "__main__":
    main()
