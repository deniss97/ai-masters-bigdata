SET mapreduce.reduce.memory.mb=4096;
ADD FILE 2a.joblib;
ADD FILE projects/2a/model.py;
ADD FILE projects/2a/predict.py;

FROM (
    FROM hw2_test 
    SELECT *,
    CASE
        WHEN hw2_test.if1 IN ('', 'NULL', '\\N') THEN 0
        ELSE hw2_test.if1
    END AS if1,
    CASE
        WHEN hw2_test.if2 IN ('', 'NULL', '\\N') THEN 0
        ELSE hw2_test.if2
    END AS if2,
    CASE
        WHEN hw2_test.if3 IN ('', 'NULL', '\\N') THEN 0
        ELSE hw2_test.if3
    END AS if3,
    CASE
        WHEN hw2_test.if4 IN ('', 'NULL', '\\N') THEN 0
        ELSE hw2_test.if4
    END AS if4,
    CASE
        WHEN hw2_test.if5 IN ('', 'NULL', '\\N') THEN 0
        ELSE hw2_test.if5
    END AS if5,
    CASE
        WHEN hw2_test.if6 IN ('', 'NULL', '\\N') THEN 0
        ELSE hw2_test.if6
    END AS if6,
    CASE
        WHEN hw2_test.if7 IN ('', 'NULL', '\\N') THEN 0
        ELSE hw2_test.if7
    END AS if7,
    CASE
        WHEN hw2_test.if8 IN ('', 'NULL', '\\N') THEN 0
        ELSE hw2_test.if8
    END AS if8,
    CASE
        WHEN hw2_test.if9 IN ('', 'NULL', '\\N') THEN 0
        ELSE hw2_test.if9
    END AS if9,
    CASE
        WHEN hw2_test.if10 IN ('', 'NULL', '\\N') THEN 0
        ELSE hw2_test.if10
    END AS if10,
    CASE
        WHEN hw2_test.if11 IN ('', 'NULL', '\\N') THEN 0
        ELSE hw2_test.if11
    END AS if11,
    CASE
        WHEN hw2_test.if12 IN ('', 'NULL', '\\N') THEN 0
        ELSE hw2_test.if12
    END AS if12,
    CASE
        WHEN hw2_test.if13 IN ('', 'NULL', '\\N') THEN 0
        ELSE hw2_test.if13
    END AS if13
    WHERE hw2_test.if1 > 20 AND hw2_test.if1 < 40
) t
INSERT OVERWRITE TABLE hw2_pred
SELECT TRANSFORM (*) USING '/opt/conda/envs/dsenv/bin/python3 predict.py'
AS id, pred;
