from airflow import DAG
from airflow.operators.empty import EmptyOperator
from airflow.operators.bash import BashOperator
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator
from airflow.sensors.filesystem import FileSensor
from datetime import datetime
import os

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
}

base_dir = '{{ dag_run.conf["base_dir"] if dag_run else "" }}'

dag = DAG(
    'deniss97_dag',
    default_args=default_args,
    description='DAG for sentiment prediction using Spark and sklearn',
    schedule=None,
    start_date=datetime(2021, 1, 1),
    catchup=False,
)

start = EmptyOperator(task_id='start', dag=dag)

feature_eng_train_task = SparkSubmitOperator(
    task_id='feature_eng_train_task',
    application=f"{base_dir}/feature_eng.py",
    name='feature_engineering',
    conn_id='spark_default',
    executor_cores=1,
    executor_memory='2g',
    num_executors=2,
    conf={
        'spark.driver.extraJavaOptions': '-Djava.security.egd=file:/dev/../dev/urandom',
        'spark.yarn.appMasterEnv.PYSPARK_PYTHON': '/opt/conda/envs/dsenv/bin/python',
        'spark.yarn.executorEnv.PYSPARK_PYTHON': '/opt/conda/envs/dsenv/bin/python'
    },
    env_vars={
        'PYSPARK_PYTHON': '/opt/conda/envs/dsenv/bin/python'
    },
    application_args=["--path-in", "/datasets/amazon/amazon_extrasmall_train.json",
                      "--path-out", "deniss97_train_out"],
    packages='org.apache.spark:spark-sql-kafka-0-10_2.12:3.0.1',
    driver_memory='1g',
    spark_binary='/usr/bin/spark3-submit',
    dag=dag
)

download_train_task = BashOperator(
    task_id='download_train_task',
    bash_command=f"hdfs dfs -get /user/ubuntu/deniss97_train_out {base_dir}/deniss97_train_out_local",
    dag=dag
)

train_task = BashOperator(
    task_id='train_task',
    bash_command=f"/opt/conda/envs/dsenv/bin/python {base_dir}/sklearn_train_script.py --train-in {base_dir}/deniss97_train_out_local --sklearn-model-out {base_dir}/6.joblib",
    dag=dag
)

model_sensor = FileSensor(
    task_id='model_sensor',
    filepath=f"{base_dir}/6.joblib",
    poke_interval=300,
    timeout=1800,
    dag=dag
)


feature_eng_test_task = SparkSubmitOperator(
    task_id='feature_eng_test_task',
    application=f"{base_dir}/feature_eng.py",
    name='feature_eng_test',
    conn_id='spark_default',
    executor_cores=1,
    executor_memory='2g',
    num_executors=2,
    conf={
        'spark.driver.extraJavaOptions': '-Djava.security.egd=file:/dev/../dev/urandom',
        'spark.yarn.appMasterEnv.PYSPARK_PYTHON': '/opt/conda/envs/dsenv/bin/python',
        'spark.yarn.executorEnv.PYSPARK_PYTHON': '/opt/conda/envs/dsenv/bin/python'
    },
    env_vars={
        'PYSPARK_PYTHON': '/opt/conda/envs/dsenv/bin/python'
    },
    application_args=["--path-in", "/datasets/amazon/amazon_extrasmall_test.json",
                      "--path-out", "deniss97_test_out"],
    packages='org.apache.spark:spark-sql-kafka-0-10_2.12:3.0.1',
    driver_memory='1g',
    spark_binary='/usr/bin/spark3-submit',
    dag=dag
)


predict_task = SparkSubmitOperator(
    task_id='predict_task',
    application=f"{base_dir}/predict.py",
    name="make_predictions",
    conn_id='spark_default',
    executor_cores=1,
    executor_memory='4g',  # Уменьшено с 8g до 1g
    num_executors=2,
    conf={
        'spark.driver.extraJavaOptions': '-Djava.security.egd=file:/dev/../dev/urandom',
        'spark.yarn.appMasterEnv.PYSPARK_PYTHON': '/opt/conda/envs/dsenv/bin/python',
        'spark.yarn.executorEnv.PYSPARK_PYTHON': '/opt/conda/envs/dsenv/bin/python',
        'spark.yarn.preserve.staging.files': 'false',
        'spark.sql.warehouse.dir': '/user/hive/warehouse',
        'spark.driver.memory': '2g',  # Уменьшено с 4g до 1g
        'spark.executor.instances': '2',  # Количество экземпляров каждого executor'а
        'spark.executor.memoryOverhead': '512m',  # Дополнительная память над обычной
        'spark.yarn.executor.memoryOverhead': '512m',
        'spark.hadoop.validateOutputSpecs': 'false'
    },
    env_vars={
        'PYSPARK_PYTHON': '/opt/conda/envs/dsenv/bin/python',
        'HADOOP_CONF_DIR': '/etc/hadoop/conf',
        'YARN_CONF_DIR': '/etc/hadoop/conf'
    },
    application_args=["--test-in", "/user/ubuntu/deniss97_test_out",
                      "--pred-out", "deniss97_hw6_prediction",
                      "--sklearn-model-in", f"{base_dir}/6.joblib"],
    packages='org.apache.spark:spark-sql-kafka-0-10_2.12:3.0.1',
    driver_memory='2g',  # Уменьшена память драйвера
    spark_binary='/usr/bin/spark3-submit',  # Убедитесь, что используете правильный путь к исполняемому файлу Spark
    dag=dag
)


end = EmptyOperator(task_id='end', dag=dag)

start >> feature_eng_train_task >> download_train_task >> train_task >> model_sensor >> feature_eng_test_task >> predict_task >> end
