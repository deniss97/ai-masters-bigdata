SET hive.exec.compress.output=false;
SET hive.cli.print.header=true;
SET hive.resultset.use.unique.column.names=false;

SET hive.exec.scratchdir=deniss97_hiveout;
SET hive.querylog.location=deniss97_hiveout;

CREATE EXTERNAL TABLE IF NOT EXISTS deniss97_hiveout
LIKE hw2_pred
STORED AS TEXTFILE
LOCATION 'hdfs:///user/deniss97/deniss97_hiveout';

INSERT OVERWRITE TABLE deniss97_hiveout SELECT * FROM hw2_pred;
