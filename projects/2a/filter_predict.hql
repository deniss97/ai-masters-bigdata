SET mapreduce.reduce.memory.mb=4096;
ADD FILE 2a.joblib;
ADD FILE projects/2a/model.py;
ADD FILE projects/2a/predict.py;

drop table temp_hw2_test;
CREATE TEMPORARY EXTERNAL TABLE temp_hw2_test AS
SELECT
  CASE WHEN hw2_test.id IS NULL THEN 0 ELSE hw2_test.id END AS id,
  CASE WHEN hw2_test.if1 IS NULL THEN 0 ELSE hw2_test.if1 END AS if1,
  CASE WHEN hw2_test.if2 IS NULL THEN 0 ELSE hw2_test.if2 END AS if2,
  CASE WHEN hw2_test.if3 IS NULL THEN 0 ELSE hw2_test.if3 END AS if3,
  CASE WHEN hw2_test.if4 IS NULL THEN 0 ELSE hw2_test.if4 END AS if4,
  CASE WHEN hw2_test.if5 IS NULL THEN 0 ELSE hw2_test.if5 END AS if5,
  CASE WHEN hw2_test.if6 IS NULL THEN 0 ELSE hw2_test.if6 END AS if6,
  CASE WHEN hw2_test.if7 IS NULL THEN 0 ELSE hw2_test.if7 END AS if7,
  CASE WHEN hw2_test.if8 IS NULL THEN 0 ELSE hw2_test.if8 END AS if8,
  CASE WHEN hw2_test.if9 IS NULL THEN 0 ELSE hw2_test.if9 END AS if9,
  CASE WHEN hw2_test.if10 IS NULL THEN 0 ELSE hw2_test.if10 END AS if10,
  CASE WHEN hw2_test.if11 IS NULL THEN 0 ELSE hw2_test.if11 END AS if11,
  CASE WHEN hw2_test.if12 IS NULL THEN 0 ELSE hw2_test.if12 END AS if12,
  CASE WHEN hw2_test.if13 IS NULL THEN 0 ELSE hw2_test.if13 END AS if13
FROM hw2_test;

drop table hw2_test;

INSERT OVERWRITE TABLE hw2_pred
SELECT TRANSFORM (*) USING '/opt/conda/envs/dsenv/bin/python3 predict.py' AS id, pred
FROM temp_hw2_test 
WHERE if1 > 20 AND if1 < 40;

