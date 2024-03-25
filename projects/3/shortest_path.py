from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
import sys

def parse_edge(s):    
    user, follower = s.split("\t")    
    return (int(user), int(follower))

def step(item):    
    prev_v, prev_d, next_v = item[0], item[1][0], item[1][1]    
    prev_d = prev_d if prev_d is not None else []    
    return (next_v, prev_d + [next_v]) if len(prev_d + [next_v]) <= max_path_length else (next_v, prev_d)

def complete(item):
    v, old_d, new_d = item[0], item[1][0], item[1][1]
    if old_d is not None and len(old_d) == 0:
        return (v, new_d)
    if old_d is not None and new_d is not None:
        return (v, min(old_d, new_d, key=len))
    if old_d is None:
        return (v, new_d)
    return (v, old_d)


conf = SparkConf()
conf.set("spark.ui.port", "4099")

sc = SparkSession.builder.config(conf=conf).appName("Spark SQL").getOrCreate()

start_node, end_node, max_path_length, input_path, output_path = int(sys.argv[1]), int(sys.argv[2]), 100, sys.argv[3], sys.argv[4]
edges = sc.sparkContext.textFile(input_path).map(parse_edge).cache()
forward_edges = edges.map(lambda e: (e[1], e[0])).partitionBy(200).persist()

distances = sc.sparkContext.parallelize([(start_node, [])]).partitionBy(200)

counter = 0

while True:
    counter += 1    
#     print(f"Running BFS iteration #{counter}")
    candidates = distances.join(forward_edges, 200).map(step)
    # Log the candidates after they are generated    
#     print(f"Candidates after iteration #{counter}: {candidates.take(10)}")
    new_distances = distances.fullOuterJoin(candidates, 200).map(complete, True).persist()    
    # Log new_distances in each iteration
#     print(f"new_distances contents in iteration #{counter}: {new_distances.take(10)}")
    # Add these lines before count:
#     print(f"Number of distances after iteration #{counter}: {distances.count()}")
#     print(f"Number of new_distances after iteration #{counter}: {new_distances.count()}")
    # Check if end_node is in any path    
    count = new_distances.filter(lambda i: end_node in i[1] if i[1] is not None else False).count()
    if count > 0 or counter > 100:
        break
    distances = new_distances

paths = new_distances.filter(lambda i: i[0] == end_node).collect()

spark = SparkSession(sc)

path_to_save = [str(start_node)] + list(map(str, paths[0][1]))

text_to_save = ','.join(path_to_save)

df = spark.sparkContext.createDataFrame([("1", text_to_save)], ["index", "path"])
path_to_save

df.select("path").write.format("text").option("header", "false").mode('overwrite').save(output_path)

sc.stop()

