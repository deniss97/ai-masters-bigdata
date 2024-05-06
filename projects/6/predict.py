import sys
import joblib
from pyspark.sql import SparkSession
from pyspark.sql.functions import regexp_replace, lit, col, rand
from pyspark.sql.types import StructType, StructField, FloatType, LongType, DoubleType

def predict(test_data, output_path, model_path):
    spark = SparkSession.builder \
        .appName("Prediction") \
        .config("spark.executor.memoryOverhead", "512m") \
        .getOrCreate()
    spark.sparkContext.setLogLevel('WARN')

    # Чтение и предобработка данных
    df_test = spark.read.json(test_data)
    print("Схема данных после загрузки:")
    df_test.printSchema()

    # Проверка наличия и обработка столбца 'vote'
    if 'vote' not in df_test.columns:
        # Генерация случайных значений для 'vote', если столбец отсутствует
        df_test = df_test.withColumn('vote', (rand() * 100).cast(DoubleType()))  # Пример с диапазоном от 0 до 100
    else:
        df_test = df_test.withColumn("vote", regexp_replace(col("vote"), ",", "").cast(DoubleType()))

    df_test = df_test.fillna({'vote': 0.0})

    # Загрузка модели
    model = joblib.load(model_path)

    # Выбираем "vote" для предсказания
    df_test = df_test.withColumn("prediction", lit(model.predict([[x] for x in df_test.select("vote").toPandas()['vote'].tolist()]).flatten()[0]).cast(DoubleType()))
    df_test = df_test.withColumn("id", df_test["vote"].cast(LongType()))

    # Преобразование результата предсказаний и сохранение
    df_test.select("id", "prediction").coalesce(1).write.csv(output_path, mode="overwrite", header=False)

    spark.stop()

if __name__ == "__main__":
    input_arg_index = sys.argv.index("--test-in") + 1
    output_arg_index = sys.argv.index("--pred-out") + 1
    model_arg_index = sys.argv.index("--sklearn-model-in") + 1

    input_path = sys.argv[input_arg_index]
    output_path = sys.argv[output_arg_index]
    model_path = sys.argv[model_arg_index]

    predict(input_path, output_path, model_path)

