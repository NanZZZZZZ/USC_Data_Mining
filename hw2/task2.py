from pyspark.context import SparkContext
from pyspark.storagelevel import StorageLevel
import itertools
import sys
import time

def frequentItemsets(tbaskets, p, prefrequent, s):
    candidates = []
    countTable = {}
    freqItems = []

    if p == 1:
        for b in tbaskets:
            for i in b:
                countTable.setdefault(i, 0)
                countTable[i] += 1
        for v in countTable:
            if countTable[v] >= s:
                freqItems.append((v))
        return sorted(freqItems)

    elif p == 2:
        for c in itertools.combinations(prefrequent, 2):
            candidates.append(c)

    else:
        for item in itertools.combinations(prefrequent, 2):
            if len(set(item[0]).intersection(set(item[1]))) == p - 2:

                can = tuple(sorted(set(item[0]).union(set(item[1]))))

                if can in candidates:
                    continue
                else:
                    temp = itertools.combinations(can, p - 1)
                    if set(temp).issubset(prefrequent):
                        candidates.append(can)

    for c in candidates:
        sc = set(c)
        countTable[c] = sum(1 for b in tbaskets if sc.issubset(b))

    for v in countTable:
        if countTable[v] >= s:
            freqItems.append(v)

    return sorted(freqItems)


def apriori(bas):
    tempbaskets = []
    for b in bas:
        tempbaskets.append(set(b[1]))

    n = len(tempbaskets)
    s = (n * S) / N

    pass_mark = 1
    freqItems = frequentItemsets(tempbaskets, pass_mark, None, s)
    if len(freqItems) == 0:return None
    preItems = freqItems
    pass_mark += 1

    while True:
        preItems = frequentItemsets(tempbaskets, pass_mark, preItems, s)
        if len(preItems) == 0:
            break
        for v in preItems:
            freqItems.append(v)
        pass_mark += 1

    return freqItems


def alterStr(object):
    if type(object)==str:
        return (object,)
    return object


def countTimes(bass):
    tempbaskets = []
    for b in bass:
        tempbaskets.append(set(b[1]))

    temp = {}

    for ca in acfi:
        sca = set(ca)
        temp[ca] = sum(1 for b in tempbaskets if sca.issubset(b))

    list = [[0 for i in range(2)] for j in range(len(temp))]
    i = 0
    for v in temp:
        list[i][0] = (v)
        list[i][1] = (temp[v])
        i += 1

    return list


def output(rdd):
    out_s = ""
    output_mark = 1
    exit_mark = True
    while exit_mark:
        if output_mark == 1:
            output_rdd = rdd.filter(lambda x: len(x) == output_mark).map(
                lambda x: (str(x).replace("',)", "')")))
            for v in output_rdd.collect():
                out_s += v
            out_s += "\n"
            output_mark += 1
        else:
            output_list = rdd.filter(lambda x: len(x) == output_mark).collect()
            if (len(output_list) == 0):
                exit_mark = False
            else:
                out_s += "\n"
                for v in output_list:
                    out_s += str(v)
                out_s += "\n"
                output_mark += 1
    out_s = out_s.strip("\n")
    return out_s




if __name__ == "__main__":

    time1 = time.time()

    sc = SparkContext( 'local[*]', 'inf553_hw2_2' )  # initiate
    sc.setLogLevel("OFF")

    T = int(sys.argv[1])
    S = int(sys.argv[2])
    input_file = sc.textFile( sys.argv[3] )  # readfile
    output_file = sys.argv[4]

    data = input_file.map(lambda x: x.split(',')).filter(lambda x: x[0] != "user_id")
    # create basket
    baskets = data.groupByKey().map(lambda x: (x[0], list(x[1]))).filter(lambda x: len(x[1]) > T).persist(StorageLevel(True, True, False, False))

    # baskets = baskets.coalesce(4, True).persist(StorageLevel(True, True, False, False))

    N = baskets.count()

    # Pass 1
    # Pass 1 Map
    can_freq_is = baskets.mapPartitions(apriori)

    # Pass 1 Reduce
    all_can_freq_is = can_freq_is.distinct().map(lambda x: alterStr(x)).sortBy(lambda x:x).sortBy(lambda x:len(x))

    if baskets.getNumPartitions() == 1:
        results = all_can_freq_is
    else:
        # Pass 2
        acfi = all_can_freq_is.collect()

        # Pass 2 Map
        can_counts = baskets.mapPartitions(countTimes)

        # Pass 2 Reduce
        results = can_counts.reduceByKey(lambda x,y: x+y).filter(lambda x: x[1] >= S).map(lambda x: x[0]).sortBy(lambda x:x).sortBy(lambda x:len(x)).persist(StorageLevel(True, True, False, False))

    output_str = "Candidates:\n"
    output_str += output(all_can_freq_is)

    output_str += "\n\nFrequent Itemsets:\n"
    output_str += output(results)

    output_str = output_str.replace(")(", "),(")

    ans_file = open(output_file, 'w')
    ans_file.write(output_str)
    ans_file.close()

    time2 = time.time()
    print("Duration: "+str(time2-time1))
