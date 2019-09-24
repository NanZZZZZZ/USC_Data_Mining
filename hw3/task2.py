from pyspark.context import SparkContext, SparkConf
from pyspark.storagelevel import StorageLevel
import sys
import time
import math
from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel, Rating


#=======================================

def model_based():

    train_data = sc.textFile(train_file).map(lambda x: x.split(',')).filter(lambda x: x[0] != "user_id").persist()
    test_data = sc.textFile(val_file).map(lambda x: x.split(',')).filter(lambda x: x[0] != "user_id").persist()

    all_data = sc.union([train_data, test_data])

    part_users = train_data.map(lambda x: x[0]).distinct().collect()
    users = all_data.map(lambda a: a[0]).distinct().collect()

    part_businesses = train_data.map(lambda x: x[1]).distinct().collect()
    businesses = all_data.map(lambda a: a[1]).distinct().collect()

    non_users = set(users).difference(set(part_users))
    non_businesses = set(businesses).difference(set(part_businesses))

    R = len(users)
    C = len(businesses)

    d_users = {}
    d_users_reverse = {}
    for u in range(0, R):
        d_users[users[u]] = u
        d_users_reverse[u] = users[u]

    d_businesses = {}
    d_businesses_reverse = {}
    for b in range(0, C):
        d_businesses[businesses[b]] = b
        d_businesses_reverse[b] = businesses[b]

    ratings = train_data.map(lambda l: Rating(d_users[l[0]], d_businesses[l[1]], float(l[2])))

    rank = 3
    numIterations = 6
    model = ALS.train(ratings, rank, numIterations, 0.03)

    testdata = test_data.map(lambda p: ( d_users[p[0]], d_businesses[p[1]] ))
    predictions = model.predictAll(testdata).persist()

    pre_ans = predictions.map(lambda x: ( d_users_reverse[x[0]], d_businesses_reverse[x[1]], x[2]))

    non_u_ans = test_data.filter(lambda x: non_users.issuperset((x[0],)) )
    non_b_ans = test_data.filter(lambda x: non_businesses.issuperset((x[1],)) )

    ans_file = open(output_file, 'w')
    ans_file.write("user_id, business_id, prediction\n")
    for c in pre_ans.collect():
        tempc = c[2]
        if c[2] > 5.0: tempc = 5.0
        elif c[2] < 1.0: tempc = 1.0
        ans_file.write(c[0] + "," + c[1] + "," + str(tempc) + "\n")
    for c in non_u_ans.collect():
        ans_file.write(c[0] + "," + c[1] + ",3.75\n")
    for c in non_b_ans.collect():
        ans_file.write(c[0] + "," + c[1] + ",3.75\n")
    ans_file.close()

    return


#=======================================

def trans(x):
    tdict = {}
    for i in range(len(x[1][0])):
        tdict[x[1][0][i]] = float(x[1][1][i])
    return (x[0], tdict)


def predict(x, dum, dbu, businesses_dict):

    b_4_p = x[1]
    u_4_p = x[0]
    pans = 0.0

    if dum.__contains__(u_4_p) == False:
        return 3.7511703308515445

    if dbu.__contains__(b_4_p) == False:
        return sum(dum[u_4_p].values()) / len(dum[u_4_p])

    rated_by_users = dbu[b_4_p]

    weights = []
    u_info = dum[u_4_p]

    for u in rated_by_users:
        u_info_2 = dum[u]

        co_rated = set(u_info_2.keys()).intersection(set(u_info.keys()))

        if len(co_rated) > 8:

            a = 0.0
            b1 = 0.0
            b2 = 0.0
            avg_r = 0.0
            avg_r2 = 0.0

            for b in co_rated:
                avg_r += u_info[b]
                avg_r2 += u_info_2[b]
            avg_r /= len(co_rated)
            avg_r2 /= len(co_rated)

            for b in co_rated:
                a += (u_info[b] - avg_r) * (u_info_2[b] - avg_r2)
                b1 += (u_info[b] - avg_r)**2
                b2 += (u_info_2[b] - avg_r2)**2

            if (b1!=0 and b2!=0):
                weights.append( (u, a/( math.sqrt(b1 * b2) ), avg_r2) )
            else:
                weights.append((u, 0.0, avg_r2))

    weights.sort(key=lambda x: x[1], reverse=True)

    wr = 0.0
    ws = 0.0
    for i in range( min(PASS, len(weights)) ):
        if weights[i][1] < 0.2:
            break
        wr += weights[i][1] * (dum[weights[i][0]][businesses_dict[b_4_p]] - weights[i][2])
        ws += math.fabs(weights[i][1])

    avg_u = sum(u_info.values()) / len(u_info)

    if (ws != 0):
        pans += avg_u + wr/ws
    else:
        pans += avg_u

    if pans > 5.0:
        pans = 5.0
    elif pans < 1.0:
        pans = 1.0

    return pans


def user_based():
    input_file = sc.textFile(train_file)
    train_data = input_file.map(lambda x: x.split(',')).filter(lambda x: x[0] != "user_id").persist(
        StorageLevel(True, True, False, False))

    input_file2 = sc.textFile(val_file)
    val_data = input_file2.map(lambda x: x.split(',')).filter(lambda x: x[0] != "user_id").persist(
        StorageLevel(True, True, False, False))

    t_businesses = train_data.map(lambda a: a[1]).distinct().collect()

    ncolumns = len(t_businesses)

    businesses_dict = {}
    for u in range(0, ncolumns):
        businesses_dict[t_businesses[u]] = u

    t_characteristic_matrix = train_data.map(lambda x: (x[0], ([businesses_dict[x[1]]], [x[2]]))).reduceByKey(
        lambda x, y: (x[0] + y[0], x[1] + y[1])).map(lambda x: trans(x))

    dum = {}
    for u in t_characteristic_matrix.collect():
        dum[u[0]] = u[1]

    businesses_users = train_data.map(lambda x: (x[1], [x[0]])).reduceByKey(lambda x, y: x + y)
    dbu = {}
    for bu in businesses_users.collect():
        dbu[bu[0]] = bu[1]

    pres = val_data.map(lambda x: (x[0], x[1], predict((x[0], x[1]), dum, dbu, businesses_dict)))

    ans_file = open(output_file, 'w')
    ans_file.write("user_id, business_id, prediction\n")
    for c in pres.collect():
        ans_file.write(c[0] + "," + c[1] + "," + str(c[2]) + "\n")
    ans_file.close()
    return


#=======================================

def i_predict(x, dum, dim, businesses_dict):

    u_4_p = x[0]
    if dum.__contains__(u_4_p) == False:
        return 3.7511703308515445

    if businesses_dict.__contains__(x[1]) == False:
        ans = sum(dum[u_4_p].values()) / len(dum[u_4_p])
        return ans

    b_4_p = businesses_dict[x[1]]
    pans = 0.0

    weights = []
    u_info = dum[u_4_p]
    i_info = dim[b_4_p]

    for i in u_info.keys():

        co_rated = set(dim[i].keys()).intersection(set(i_info.keys()))
        t = (i, b_4_p)
        t = tuple(sorted(t))

        if item_weights.__contains__(t):
            if item_weights[t] > 0.2:
                weights.append((i, item_weights[t]))
            continue
        else:
            if len(co_rated) >= 12:
                i_info_2 = dim[i]

                a = 0.0
                b1 = 0.0
                b2 = 0.0
                avg_r = 0.0
                avg_r2 = 0.0

                for b in co_rated:
                    avg_r += i_info[b]
                    avg_r2 += i_info_2[b]
                avg_r /= len(co_rated)
                avg_r2 /= len(co_rated)


                for b in co_rated:
                    a += (i_info[b]-avg_r) * (i_info_2[b]-avg_r2)
                    b1 += (i_info[b] - avg_r)**2
                    b2 += (i_info_2[b] - avg_r2)**2

                if (b1!=0 and b2!=0):
                    w = a/( math.sqrt(b1 * b2))
                    weights.append( (i, w) )
                    item_weights[t] = w
                else:
                    item_weights[t] = 0.0
            else:
                item_weights[t] = 0.0

    if(len(weights) < 2):
        return sum(u_info.values()) / len(u_info)

    weights.sort(key=lambda x: x[1], reverse=True)

    wr = 0.0
    ws = 0.0

    for i in range( min(PASS2, len(weights)) ):
        if weights[i][1] < 0.2:
            break
        wr += weights[i][1] * dum[u_4_p][weights[i][0]]
        ws += math.fabs(weights[i][1])

    if (ws != 0):
        pans +=  wr/ws
    else:
        pans += (sum(u_info.values()) / len(u_info))

    if pans > 5.0:
        pans = 5.0
    elif pans < 1.0:
        pans = 1.0

    return pans

def item_based():
    input_file = sc.textFile(train_file)
    train_data = input_file.map(lambda x: x.split(',')).filter(lambda x: x[0] != "user_id").persist(
        StorageLevel(True, True, False, False))

    input_file2 = sc.textFile(val_file)
    val_data = input_file2.map(lambda x: x.split(',')).filter(lambda x: x[0] != "user_id").persist(
        StorageLevel(True, True, False, False))

    t_users = train_data.map(lambda a: a[0]).distinct().collect()
    t_businesses = train_data.map(lambda a: a[1]).distinct().collect()
    R = len(t_users)
    C = len(t_businesses)

    users_dict = {}
    for u in range(0, R):
        users_dict[t_users[u]] = u

    businesses_dict = {}
    for u in range(0, C):
        businesses_dict[t_businesses[u]] = u

    t_characteristic_matrix = train_data.map(lambda x: (x[0], ([businesses_dict[x[1]]], [x[2]]))).reduceByKey(
        lambda x, y: (x[0] + y[0], x[1] + y[1])).persist(StorageLevel(True, True, False, False))
    t2 = t_characteristic_matrix.map(lambda x: trans(x))
    dum = {}
    for u in t2.collect():
        dum[u[0]] = u[1]

    ti_characteristic_matrix = train_data.map(lambda x: (businesses_dict[x[1]], ([users_dict[x[0]]], [x[2]]))).reduceByKey(
        lambda x, y: (x[0] + y[0], x[1] + y[1])).persist(StorageLevel(True, True, False, False))
    ti2 = ti_characteristic_matrix.map(lambda x: trans(x))
    dim = {}
    for u in ti2.collect():
        dim[u[0]] = u[1]

    pres = val_data.map(lambda x: (x[0], x[1], i_predict((x[0], x[1], x[2]), dum, dim, businesses_dict))).persist(StorageLevel(True, True, False, False))

    ans_file = open(output_file, 'w')
    ans_file.write("user_id, business_id, prediction\n")
    for c in pres.collect():
        ans_file.write(c[0]+","+c[1]+","+str(c[2])+"\n")
    ans_file.close()

    return


#=======================================


if __name__ == "__main__":

    time1 = time.time()

    PASS = 20
    PASS2 = 20

    item_weights = {}

    conf = SparkConf().setAppName('inf553_hw3_2').setMaster('local[*]')
    sc = SparkContext(conf = conf)
    sc.setLogLevel("OFF")

    train_file = sys.argv[1]
    val_file = sys.argv[2]
    case_mark = int(sys.argv[3])
    output_file = sys.argv[4]


    if case_mark == 1:
        model_based()
    elif case_mark == 2:
        user_based()
    elif case_mark == 3:
        item_based()
    else:
        exit(1)


    time2 = time.time()
    print("Duration: " + str(time2 - time1))
