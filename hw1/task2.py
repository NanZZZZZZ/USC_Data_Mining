from pyspark.context import SparkContext
from pyspark.storagelevel import StorageLevel
import time
import json
import sys

sc = SparkContext( 'local[*]', 'inf553_hw1_2' )

input_file_review = sc.textFile( sys.argv[1] )
data_review = input_file_review.map(lambda a:json.loads(a)).map(lambda a: (a['business_id'], a['stars']))

input_file_business = sc.textFile( sys.argv[2] )
data_business = input_file_business.map(lambda a:json.loads(a)).map(lambda a: (a['business_id'], a['state'] ) )

data = data_review.join(data_business)

mstatestar = data.map(lambda x: (x[1][1], (x[1][0], 1))).reduceByKey(lambda x, y: (x[0]+y[0], x[1]+y[1])).sortByKey()

mstateavgstar = mstatestar.map(lambda x: (x[0], format(float(x[1][0])/x[1][1])) ).sortBy(lambda x: x[1], ascending=False).persist(StorageLevel(True, True, False, False))

# 1-collect
begin_time_1 = time.time()

m1 = mstateavgstar.collect()
for i in range(5):
    print(m1[i])

end_time_1 = time.time()

# 2-take
begin_time_2 = time.time()

m2 = mstateavgstar.take(5)
print(m2)

end_time_2 = time.time()

ans_file1=open( sys.argv[3], 'w')
ans_file1.write("state,stars")
for i in range(mstateavgstar.count()):
    ans_file1.write("\n"+m1[i][0]+","+m1[i][1])
ans_file1.close()


ans2_dict = {"m1": end_time_1-begin_time_1,
             "m2": end_time_2-begin_time_2,
             "explanation": "The Method 2 needs less time, because the quantities of the data need to be dealed with are different. In M1, collect function transfers all the data, but in M2, the take function only takes first 5 and finishes the transfermation."}

ans2_json = json.dumps(ans2_dict)

ans_file2=open( sys.argv[4], 'w')
ans_file2.write(ans2_json)
ans_file2.close()
