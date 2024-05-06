import sys
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, FloatType

def copy_data(input_path, output_path):
    # Создание Spark сессии
    spark = SparkSession.builder.appName("DataCopying").getOrCreate()
    spark.sparkContext.setLogLevel('WARN')

    # Определение схемы данных
    schema = StructType([
        StructField("id", StringType()),
        StructField("pred", FloatType())
    ])

    # Чтение данных с использованием заданной схемы
    df_test = spark.read.schema(schema).csv(input_path)
    df_test.cache()  # Кэширование DataFrame

    # Сохранение данных в CSV файл без заголовков в указанном пути
    # Здесь важно правильно настроить параметры, если формат исходного файла отличается
    df_test.write.csv(output_path, mode='overwrite', header=False)

    # Завершение сессии Spark
    spark.stop()

if __name__ == "__main__":
    input_arg_index = sys.argv.index("--test-in") + 1
    output_arg_index = sys.argv.index("--pred-out") + 1

    input_path = sys.argv[input_arg_index]
    output_path = sys.argv[output_arg_index]

    copy_data(input_path, output_path)

