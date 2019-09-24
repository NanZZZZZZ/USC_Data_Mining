from pyspark.context import SparkContext, SparkConf
from pyspark.storagelevel import StorageLevel
import itertools
import sys
import time
import random

def generate_random(x):
    a = []
    while x > 0:
        a.append( random.randint(0, nrows) )
        x -= 1
    return a

def minhash(characs):
    sig = [min((ai * c) % nrows for c in characs) for ai in a]
    return sig

def lsh(x):
    N = len(a)
    R = 2
    B = int(N / R)

    rlist = []
    for i in range(B):
        tl = [ x[1][a] for a in range(i*R, (i+1)*R) ]
        rlist.append( ( (i, tuple(tl)), [x[0]] ) )
    return rlist

def generate_cans(can):
    cans = []
    for c in itertools.combinations(can, 2):
        cans.append(tuple(sorted(c)))
    return cans

def jaccard_similarity(pairs):
    global d_characteristic_matrix
    s1 = d_characteristic_matrix[pairs[0]]
    s2 = d_characteristic_matrix[pairs[1]]
    inter = s1.intersection(s2)
    un = s1.union(s2)
    sim = float(len(inter)/len(un))
    return (pairs[0], pairs[1], sim)


if __name__ == "__main__":

    time1 = time.time()

    conf = SparkConf().setAppName('inf553_hw3_1').setMaster('local[*]')
    sc = SparkContext(conf=conf)  # initiate
    sc.setLogLevel("OFF")

    input_file = sc.textFile(sys.argv[1])  # readfile
    output_file = sys.argv[2]

    data = input_file.map(lambda x: x.split(',')).filter(lambda x: x[0] != "user_id").persist(StorageLevel(True, True, False, False))

    users = data.map(lambda a: a[0]).distinct().collect()
    nrows = len(users)

    users_dict = {}
    for u in range(0, nrows):
        users_dict[users[u]] = u

    characteristic_matrix = data.map(lambda x: (x[1], [users_dict[x[0]]])).reduceByKey(lambda x, y: x + y).persist(StorageLevel(True, True, False, False))

    d_characteristic_matrix = {}
    cm = characteristic_matrix.map(lambda x: (x[0], set(x[1]))).collect()
    for i in cm:
        d_characteristic_matrix[i[0]] = i[1]

    # a = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29,
    #      31, 37, 41, 43, 47, 53, 59, 61, 67, 71,
    #      73, 79, 83, 89, 97, 101, 103, 127, 131, 137,
    #      139, 149, 151, 157, 163, 167, 173, 179, 181, 191]
    a = generate_random(40)

    signature_matrix = characteristic_matrix.map(lambda x: (x[0], minhash(x[1])))

    can = signature_matrix.flatMap(lambda x: lsh(x)).reduceByKey(lambda x,y: x+y).filter(lambda x: len(x[1])>1).map(lambda x: x[1])
    cans = can.flatMap(lambda x: generate_cans(x)).distinct().sortByKey()

    jac_sims = cans.map(lambda x: jaccard_similarity(x)).filter(lambda x: x[2]>=0.5).sortBy(lambda x: x[1]).sortBy(lambda x: x[0])

    ans_file = open(output_file, 'w')
    ans_file.write("business_id_1, business_id_2, similarity\n")
    for c in jac_sims.collect():
        ans_file.write(c[0]+","+c[1]+","+str(c[2])+"\n")
    ans_file.close()

    print("Count: " + str(jac_sims.count()))

    time2 = time.time()
    print("Duration: " + str(time2 - time1))

