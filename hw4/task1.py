from pyspark.context import SparkContext
import itertools
import sys
import time


if __name__ == "__main__":

    time1 = time.time()

    sc = SparkContext( 'local[*]', 'inf553_hw4_1' )
    sc.setLogLevel("OFF")

    threshold = int(sys.argv[1])
    input_file = sc.textFile( sys.argv[2])
    output_file1 = sys.argv[3]

    data = input_file.map(lambda x: x.split(',')).filter(lambda x: x[0] != "user_id")

    bu = data.map(lambda x: (x[1], [x[0]])).reduceByKey(lambda x,y: x+y).map(lambda x: (x[0], sorted(x[1])))
    up = bu.flatMap(lambda x: list(itertools.combinations(x[1], 2))).map(lambda x: (x, 1)).reduceByKey(lambda x,y: x+y).filter(lambda x: x[1]>=threshold).map(lambda x: x[0])

    # print(up.collect())
    time2 = time.time()
    print("Duration: "+str(time2-time1))