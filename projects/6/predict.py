import sys
import numpy as np
import joblib
from pyspark.sql import SparkSession, Row  # Исправлено добавлением импорта Row
from pyspark.sql.functions import regexp_replace, lit
from pyspark.sql.types import FloatType

def predict(test_data, output_path, model_path):
    spark = SparkSession.builder \
        .appName("Prediction") \
        .config("spark.executor.memoryOverhead", "512m")  # Обновлено согласно рекомендации
        .getOrCreate()
    spark.sparkContext.setLogLevel('WARN')

    # Чтение и предобработка данных
    df_test = spark.read.json(test_data)
    print("Схема данных после загрузки:")
    df_test.printSchema()  # Для проверки доступных столбцов

    if 'vote' in df_test.columns:
        df_test = df_test.withColumn("vote", regexp_replace("vote", ",", "").cast(FloatType()))
    else:
        df_test = df_test.withColumn("vote", lit(0.0).cast(FloatType()))

    df_test = df_test.fillna({'vote': 0})  # Применение fillna на уровне DataFrame

    # Загрузка модели
    model = joblib.load(model_path)

    # Применение модели для каждой строки DataFrame
    pd_test = df_test.select("vote").toPandas()
    predictions = model.predict(pd_test['vote'].astype(float).values.reshape(-1, 1))

    # Обработка NaN в предсказаниях
    predictions = np.where(np.isnan(predictions), 0.5, predictions)

    # Преобразование списка предсказаний в Spark DataFrame
    rows = [Row(id=i, prediction=float(x)) for i, x in enumerate(predictions)]
    result_df = spark.createDataFrame(rows)

    # Ограничение на 4,000,000 записей
    result_df = result_df.limit(4000000)

    # Запись результатов без заголовков и индексов
    result_df.coalesce(1).write.mode('overwrite').option("header", "false").json(output_path)

    spark.stop()

if __name__ == "__main__":  # Правильное использование __name__ и __main__
    input_arg_index = sys.argv.index("--test-in") + 1
    output_arg_index = sys.argv.index("--pred-out") + 1
    model_arg_index = sys.argv.index("--sklearn-model-in") + 1

    input_path = sys.argv[input_arg_index]
    output_path = sys.argv[output_arg_index]
    model_path = sys.argv[model_arg_index]

    predict(input_path, output_path, model_path)

