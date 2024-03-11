SET mapreduce.reduce.memory.mb=4096;

ADD FILE 2a.joblib;
ADD FILE projects/2a/model.py;
ADD FILE projects/2a/predict.py;

FROM (
    FROM hw2_test 
    SELECT *
    WHERE if1 > 20 AND if1 < 40 AND if1 NOT IN ('', 'null', 'NULL', '\\N') AND if2 NOT IN ('', 'null', 'NULL', '\\N') AND if3 NOT IN ('', 'null', 'NULL', '\\N') AND if4 NOT IN ('', 'null', 'NULL', '\\N') AND if5 NOT IN ('', 'null', 'NULL', '\\N') AND if6 NOT IN ('', 'null', 'NULL', '\\N') AND if7 NOT IN ('', 'null', 'NULL', '\\N') AND if8 NOT IN ('', 'null', 'NULL', '\\N') AND if9 NOT IN ('', 'null', 'NULL', '\\N') AND if10 NOT IN ('', 'null', 'NULL', '\\N') AND if11 NOT IN ('', 'null', 'NULL', '\\N') AND if12 NOT IN ('', 'null', 'NULL', '\\N') AND if13 NOT IN ('', 'null', 'NULL', '\\N')
) t
INSERT OVERWRITE TABLE hw2_pred
SELECT TRANSFORM (*)
USING '/opt/conda/envs/dsenv/bin/python3 predict.py'
AS id, pred;
