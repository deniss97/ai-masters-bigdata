import sys
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, IntegerType, FloatType

def copy_data(input_path, output_path):
    # Создание Spark сессии
    spark = SparkSession.builder.appName("DataCopying").getOrCreate()
    spark.sparkContext.setLogLevel('WARN')
    
    # Определение схемы данных
    schema = StructType([
        StructField("id", IntegerType()),
        StructField("pred", FloatType())
    ])

    # Чтение и предобработка данных с использованием заданной схемы
    df_test = spark.read.schema(schema).json(input_path)
    df_test.cache() # Кэширование DataFrame

    # Запись данных без изменений в CSV
    df_test.limit(4000000).coalesce(1).write.mode('overwrite').option("header", "true").csv(output_path)

    # Завершение сессии Spark
    spark.stop()

if __name__ == "__main__":
    # Получение путей из аргументов командной строки
    input_arg_index = sys.argv.index("--test-in") + 1
    output_arg_index = sys.argv.index("--pred-out") + 1

    input_path = sys.argv[input_arg_index]
    output_path = sys.argv[output_arg_index]

    copy_data(input_path, output_path)

