from pyspark.context import SparkContext
import itertools
import sys
import time
import collections

def calculate_betweenness(x, d_adjacents):

    vertices_values = {}
    vertices_pathno = {}
    betweenness = {}
    stack = collections.deque()
    parent_vertices = {}

    current_level = [x]
    visited = [x]
    vertices_pathno[x] = 1
    vertices_values[x] = 1

    while len(current_level) > 0:
        temp_visited = []
        for i in current_level:
            for v in d_adjacents[i]:
                if v not in visited:
                    if v not in temp_visited:
                        temp_visited.append(v)
                        stack.append(v)
                        vertices_values.setdefault(v, 1)
                        vertices_pathno.setdefault(v, 0)
                        parent_vertices.setdefault(v, [])
                    vertices_pathno[v] += vertices_pathno[i]
                    parent_vertices[v] += [i]
        for v in temp_visited:
            visited.append(v)
        current_level = temp_visited

    while len(stack) > 0:
        current_vertex = stack.pop()
        for v in parent_vertices[current_vertex]:
            e = tuple(sorted([current_vertex, v]))
            betweenness[e] = float(vertices_values[current_vertex] * vertices_pathno[v] / vertices_pathno[current_vertex])
            vertices_values[v] += betweenness[e]

    ans = []
    for e in betweenness:
        ans.append((e, betweenness[e]))

    return ans

def get_communities(d_adjacents):
    adjacents = {}
    for i in d_adjacents:
        l = []
        for item in d_adjacents[i]:
            l.append(item)
        adjacents[i] = l

    communities = []
    group = []
    visited = []
    stack = collections.deque()

    while len(adjacents) > 0:
        first_non_visited = list(adjacents.keys())[0]
        group.clear()
        group.append(first_non_visited)
        stack.clear()
        stack.append(first_non_visited)
        visited.append(first_non_visited)
        while len(stack) > 0:
            current_vertex = stack.pop()
            for i in adjacents[current_vertex]:
                if i not in visited:
                    group.append(i)
                    stack.append(i)
                    visited.append(i)
                    adjacents[i].remove(current_vertex)

            adjacents.pop(current_vertex)
        communities.append(tuple(group))

    return communities

def calculate_modularity(current_communities, A, m):
    modularity = 0.0
    for s in current_communities:
        point_pairs = itertools.combinations(s, 2)
        for p in s:
            k = len(A[p])
            modularity += (0 - float(k * k) / (2 * m))
        for pair in point_pairs:
            Aij = 0
            if A[pair[0]].__contains__(pair[1]):
                Aij += 1
            ki = len(A[pair[0]])
            kj = len(A[pair[1]])
            modularity += 2 * (Aij - float(ki * kj)/(2 * m))
    modularity /= (2 * m)
    return modularity


if __name__ == "__main__":

    time1 = time.time()

    sc = SparkContext( 'local[*]', 'inf553_hw4_2' )
    sc.setLogLevel("OFF")

    threshold = int(sys.argv[1])
    input_file = sc.textFile( sys.argv[2])
    output_file_bet = sys.argv[3]
    output_file_com = sys.argv[4]

    data = input_file.map(lambda x: x.split(',')).filter(lambda x: x[0] != "user_id")

    bu = data.map(lambda x: (x[1], [x[0]])).reduceByKey(lambda x,y: x+y).map(lambda x: (x[0], (x[1])))
    up = bu.flatMap(lambda x: list(itertools.combinations(x[1], 2))).map(lambda x: (x, 1)).\
        reduceByKey(lambda x,y: x+y).filter(lambda x: x[1]>=threshold).\
        map(lambda x: x[0]).persist()

    vertices = up.flatMap(lambda x: x).distinct().persist()
    N = vertices.count()

    up_r = up.map(lambda x: (x[1], x[0]))
    adj = up.map(lambda x: (x[0], [x[1]])).reduceByKey(lambda x, y: x + y)
    adj_r = up_r.map(lambda x: (x[0], [x[1]])).reduceByKey(lambda x, y: x + y)
    adjacents = sc.union([adj, adj_r]).reduceByKey(lambda x,y: x+y).persist()

    d_adjacents = adjacents.collectAsMap()

    A = adjacents.collectAsMap()

    betweennesses = vertices.flatMap(lambda x: calculate_betweenness(x, d_adjacents)).\
        reduceByKey(lambda x,y: x+y).\
        map(lambda x: (x[0], x[1]/2.0)). \
        sortBy(lambda x: x[0][1]). \
        sortBy(lambda x: x[0][0]). \
        sortBy(lambda x: x[1], False). \
        persist()

    bet_file = open(output_file_bet, 'w')
    for i in betweennesses.collect():
        bet_file.write("('"+i[0][0] + "', '" + i[0][1] + "'), " + str(i[1]) + "\n")
    bet_file.close()

    m = up.count()

    betweenness_groups = betweennesses.map(lambda x: (x[1], [x[0]])).reduceByKey(lambda x,y: x+y).sortBy(lambda x: x[0], ascending=False).persist()
    modularity = -1.0
    last_modularity = -1.0
    best_communities = []

    PASS = 0

    while True:
        edges_2_remove = betweenness_groups.take(1)[0][1]

        for e in edges_2_remove:
            d_adjacents[e[0]].remove(e[1])
            d_adjacents[e[1]].remove(e[0])

        current_communities = get_communities(d_adjacents)

        current_modularity = calculate_modularity(current_communities, A, m)

        if current_modularity > modularity:
            modularity = current_modularity
            best_communities = current_communities

        last_modularity = current_modularity
        PASS += 1

        betweennesses = vertices.flatMap(lambda x: calculate_betweenness(x, d_adjacents)). \
            reduceByKey(lambda x, y: x + y). \
            map(lambda x: (x[0], float(x[1] / 2))). \
            sortBy(lambda x: x[0][1]). \
            sortBy(lambda x: x[0][0]). \
            sortBy(lambda x: x[1], False). \
            persist()
        betweenness_groups = betweennesses.map(lambda x: (x[1], [x[0]])).reduceByKey(lambda x, y: x + y).sortBy(lambda x: x[0], ascending=False).persist()

        if betweenness_groups.count()==0:
            break

    communities_rdd = sc.parallelize(best_communities).map(lambda x: sorted(x)).sortBy(lambda x: x).sortBy(lambda x: len(x)).persist()

    com_file = open(output_file_com, 'w')
    ans_communities = communities_rdd.collect()
    for s in ans_communities:
        s = str(s).replace("[", "").replace("]", "")
        com_file.write(s + "\n")
    com_file.close()

    print(communities_rdd.count())
    print(modularity)
    print(PASS)

    time2 = time.time()
    print("Duration: "+str(time2-time1))