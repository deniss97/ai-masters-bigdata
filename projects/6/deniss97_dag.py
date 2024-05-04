from airflow import DAG
from airflow.operators.dummy import DummyOperator
from airflow.operators.bash import BashOperator
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator
from datetime import datetime

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
}

dag = DAG(
    'deniss97_dag',
    default_args=default_args,
    description='DAG for sentiment prediction using Spark and sklearn',
    schedule_interval=None,
    start_date=datetime(2021, 1, 1),
    catchup=False,
)

start = DummyOperator(
    task_id='start',
    dag=dag
)

feature_eng_train_task = SparkSubmitOperator(
    task_id='feature_eng_train_task',
    application="{{ var.value.base_dir }}/feature_eng.py",
    name="feature_eng_train",
    application_args=["--path-in", "/datasets/amazon/amazon_extrasmall_train.json",
                      "--path-out", "{{ var.value.base_dir }}/deniss97_train_out"],
    dag=dag,
)

download_train_task = BashOperator(
    task_id='download_train_task',
    bash_command='hdfs dfs -get /user/deniss97/deniss97_train_out {{ var.value.base_dir }}/deniss97_train_out_local',
    dag=dag
)

train_task = SparkSubmitOperator(
    task_id='train_task',
    application="{{ var.value.base_dir }}/train_model.py",
    name="train_model",
    application_args=["--train-in", "{{ var.value.base_dir }}/deniss97_train_out_local",
                      "--sklearn-model-out", "{{ var.value.base_dir }}/6.joblib"],
    dag=dag,
)

model_sensor = FileSensor(
    task_id='model_sensor',
    filepath='{{ var.value.base_dir }}/6.joblib',
    poke_interval=60,
    timeout=600,
    dag=dag
)

feature_eng_test_task = SparkSubmitOperator(
    task_id='feature_eng_test_task',
    application="{{ var.value.base_dir }}/feature_eng.py",
    name="feature_eng_test",
    application_args=["--path-in", "/datasets/amazon/amazon_extrasmall_test.json",
                      "--path-out", "{{ var.value.base_dir }}/deniss97_test_out"],
    dag=dag,
)

predict_task = SparkSubmitOperator(
    task_id='predict_task',
    application="{{ var.value.base_dir }}/predict.py",
    name="make_predictions",
    application_args=["--test-in", "{{ var.value.base_dir }}/deniss97_test_out",
                      "--pred-out", "/user/deniss97/deniss97_hw6_prediction",
                      "--sklearn-model-in", "{{ var.value.base_dir }}/6.joblib"],
    dag=dag,
)

end = DummyOperator(
    task_id='end',
    dag=dag
)

start >> feature_eng_train_task >> download_train_task >> train_task >> model_sensor >> feature_eng_test_task >> predict_task >> end

