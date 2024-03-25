from pyspark import SparkConf, SparkContext
import sys

def parse_edge(s):
    user, follower = s.split("\t")
    return (int(user), int(follower))

def step(item):
    prev_v, prev_d, next_v = item[0], item[1][0], item[1][1]
    return (next_v, prev_d + [next_v]) if len(prev_d + [next_v]) <= max_path_length else (next_v, [])

def complete(item):
    v, old_d, new_d = item[0], item[1][0], item[1][1]
    return (v, new_d if old_d is None else old_d)

start_node, end_node, input_path, output_path = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3], sys.argv[4]
max_path_length = 100
conf = SparkConf().setAppName('ShortestPathBFS')
sc = SparkContext(conf=conf)

edges = sc.textFile(input_path).map(parse_edge).cache()
forward_edges = edges.map(lambda e: (e[1], e[0])).partitionBy(200).persist()

distances = sc.parallelize([(start_node, [start_node])]).partitionBy(200)
while True:
    candidates = distances.join(forward_edges,200).map(step)
    new_distances = distances.fullOuterJoin(candidates,200).map(complete, True).persist()
    count = new_distances.filter(lambda i: len(i[1]) == 0).count()
    if count > 0:
        break
    distances = new_distances

paths = new_distances.filter(lambda i: i[1] == end_node).collect()
for path in paths:
    print(','.join(map(str, path[1])))

sc.stop()

