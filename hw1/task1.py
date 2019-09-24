from pyspark.context import SparkContext
from pyspark.storagelevel import StorageLevel
import json
import sys

sc = SparkContext( 'local[*]', 'inf553_hw1_1' )  # initiate

input_file = sc.textFile( sys.argv[1] )  # readfile
data = input_file.map( lambda x: json.loads(x) ).map( lambda x: (x['review_id'], (x['user_id'], x['business_id'], x['useful'], x['stars'], len(x['text'])))).persist(StorageLevel(True, True, False, False)) # deal with json

museful = data.filter( lambda x: x[1][2] > 0 ).count()

mfivestar = data.filter( lambda x: x[1][3] == 5.0 ).count()

mlongestreview = data.map( lambda x: (x[1][4], 1) ).top(1)

muser = data.map(lambda x: (x[1][0], 1) ).reduceByKey(lambda x,y: x+y).sortByKey().persist(StorageLevel(True, True, False, False))

musernum = muser.count()

muserreview = muser.takeOrdered(20, lambda x: -x[1])

mbusiness = data.map(lambda x: ((x[1][1]), 1) ).reduceByKey(lambda x,y: x+y).sortByKey().persist(StorageLevel(True, True, False, False))

mbusinessnum = mbusiness.count()

mbusinessreview = mbusiness.takeOrdered(20, lambda x: -x[1])

ans_dict = {"n_review_useful": museful,
             "n_review_5_star": mfivestar,
             "n_characters": mlongestreview[0][0],
             "n_user": musernum,
             "top20_user": muserreview,
             "n_business": mbusinessnum,
             "top20_business": mbusinessreview}

ans_json = json.dumps(ans_dict)

ans_file=open( sys.argv[2], 'w')
ans_file.write(ans_json)
ans_file.close()
