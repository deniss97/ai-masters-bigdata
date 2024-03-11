SET mapreduce.reduce.memory.mb=4096;

ADD FILE 2a.joblib;
ADD FILE projects/2a/model.py;
ADD FILE projects/2a/predict.py;

FROM (
    FROM hw2_test
    SELECT *,
    CASE
        WHEN if1 IN ('', 'NULL', '\\N') THEN 0
        ELSE if1
    END AS if1_filtered,
    CASE
        WHEN if2 IN ('', 'NULL', '\\N') THEN 0
        ELSE if2
    END AS if2_filtered
    CASE
        WHEN if3 IN ('', 'NULL', '\\N') THEN 0
        ELSE if3
    END AS if3_filtered
    CASE
        WHEN if4 IN ('', 'NULL', '\\N') THEN 0
        ELSE if4
    END AS if4_filtered
    CASE
        WHEN if5 IN ('', 'NULL', '\\N') THEN 0
        ELSE if5
    END AS if5_filtered
    CASE
        WHEN if6 IN ('', 'NULL', '\\N') THEN 0
        ELSE if6
    END AS if6_filtered
    CASE
        WHEN if7 IN ('', 'NULL', '\\N') THEN 0
        ELSE if7
    END AS if7_filtered
    CASE
        WHEN if8 IN ('', 'NULL', '\\N') THEN 0
        ELSE if8
    END AS if8_filtered
    CASE
        WHEN if9 IN ('', 'NULL', '\\N') THEN 0
        ELSE if9
    END AS if9_filtered
    CASE
        WHEN if10 IN ('', 'NULL', '\\N') THEN 0
        ELSE if10
    END AS if10_filtered
    CASE
        WHEN if11 IN ('', 'NULL', '\\N') THEN 0
        ELSE if11
    END AS if11_filtered
    CASE
        WHEN if12 IN ('', 'NULL', '\\N') THEN 0
        ELSE if12
    END AS if12_filtered
    CASE
        WHEN if13 IN ('', 'NULL', '\\N') THEN 0
        ELSE if13
    END AS if13_filtered
    WHERE if1 > 20 AND if1 < 40
) t
INSERT OVERWRITE TABLE hw2_pred
SELECT TRANSFORM (*) USING '/opt/conda/envs/dsenv/bin/python3 predict.py'
AS id, pred;

