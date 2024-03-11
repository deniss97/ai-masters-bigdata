SET mapreduce.reduce.memory.mb=4096;

ADD FILE 2a.joblib;
ADD FILE projects/2a/model.py;
ADD FILE projects/2a/predict.py;

FROM (
    FROM hw2_test 
    SELECT *
    WHERE if1 > 20 AND if1 < 40 AND if1 NOT IN ('', 'null', 'NULL', '\\N') AND if2 NOT IN ('', 'null', 'NULL', '\\N') AND if3 NOT IN ('', 'null', 'NULL', '\\N') AND if4 NOT IN ('', 'null', 'NULL', '\\N') AND if5 NOT IN ('', 'null', 'NULL', '\\N') AND if6 NOT IN ('', 'null', 'NULL', '\\N') AND if7 NOT IN ('', 'null', 'NULL', '\\N') AND if8 NOT IN ('', 'null', 'NULL', '\\N') AND if9 NOT IN ('', 'null', 'NULL', '\\N') AND if10 NOT IN ('', 'null', 'NULL', '\\N') AND if11 NOT IN ('', 'null', 'NULL', '\\N') AND if12 NOT IN ('', 'null', 'NULL', '\\N') AND if13 NOT IN ('', 'null', 'NULL', '\\N') AND cf1 NOT IN ('', 'null', 'NULL', '\\N') AND cf2 NOT IN ('null', '', 'NULL', '\\N') AND cf3 NOT IN ('null', '', 'NULL', '\\N') AND cf4 NOT IN ('null', '', 'NULL', '\\N') AND cf5 NOT IN ('null', '', 'NULL', '\\N') AND cf6 NOT IN ('null', '', 'NULL', '\\N') AND cf7 NOT IN ('null', '', 'NULL', '\\N') AND cf8 NOT IN ('null', '', 'NULL', '\\N') AND cf9 NOT IN ('null', '', 'NULL', '\\N') AND cf10 NOT IN ('null', '', 'NULL', '\\N') AND cf11 NOT IN ('null', '', 'NULL', '\\N') AND cf12 NOT IN ('null', '', 'NULL', '\\N') AND cf13 NOT IN ('null', '', 'NULL', '\\N') AND cf14 NOT IN ('null', '', 'NULL', '\\N') AND cf15 NOT IN ('null', '', 'NULL', '\\N') AND cf16 NOT IN ('null', '', 'NULL', '\\N') AND cf17 NOT IN ('null', '', 'NULL', '\\N') AND cf18 NOT IN ('null', '', 'NULL', '\\N') AND cf19 NOT IN ('null', '', 'NULL', '\\N') AND cf20 NOT IN ('null', '', 'NULL', '\\N') AND cf21 NOT IN ('null', '', 'NULL', '\\N') AND cf22 NOT IN ('null', '', 'NULL', '\\N') AND cf23 NOT IN ('null', '', 'NULL', '\\N') AND cf24 NOT IN ('null', '', 'NULL', '\\N') AND cf25 NOT IN ('null', '', 'NULL', '\\N') AND cf26 NOT IN ('null', '', 'NULL', '\\N')
) t
INSERT OVERWRITE TABLE hw2_pred
SELECT TRANSFORM (*)
USING '/opt/conda/envs/dsenv/bin/python3 predict.py'
AS id, pred;
