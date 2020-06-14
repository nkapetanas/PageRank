from collections import defaultdict

import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt

TIME_PRINT = "--- %s seconds ---"


def page_rank(graph, num_iters):
    # initialize a dictionary with node value and a unit of importance
    pages_with_score = {node: 1 for node in graph.keys()}

    for _ in range(num_iters):
        old_pages_with_score = dict(pages_with_score)
        keys_empty = list()

        for page, score in pages_with_score.items():
            connections = graph[page]
            new_score = 0
            for connection in connections:
                outgoing_edges_from_node = len(graph[connection])
                try:
                    new_score += old_pages_with_score[connection] / outgoing_edges_from_node
                except KeyError:
                    keys_empty.append(connection)

            pages_with_score[page] = new_score
    return pages_with_score


def page_rank_improved(graph, a, num_iters):
    # initialize a dictionary with node value and a unit of importance
    pages_with_score = {node: 1 for node in graph.keys()}

    norm_part = (1 - a) * np.ones(len(graph.keys()))

    for _ in range(num_iters):
        old_pages_with_score = dict(pages_with_score)

        for page, score in pages_with_score.items():
            connections = graph[page]
            new_score = 0
            for connection in connections:
                new_score += old_pages_with_score[connection] / len(graph[connection])

            new_score = new_score * a + norm_part[0]
            pages_with_score[page] = new_score
    return pages_with_score


def get_n_elements_dict(n, reversed_list, sorted_dict):
    if reversed_list:
        for key in list(reversed(list(sorted_dict)))[0:n]:
            print("key {}, value {} ".format(key, sorted_dict[key]))
        return True

    for key in list(sorted_dict)[0:n]:
        print("key {}, value {} ".format(key, sorted_dict[key]))


def compute_page_rank(graph, iter_number):
    rankings = page_rank(graph, iter_number)

    for k, v in rankings.items():
        rankings[k] = round(v, 2)

    rankings_sorted = {k: v for k, v in sorted(rankings.items(), key=lambda item: item[1])}

    get_n_elements_dict(20, False, rankings_sorted)
    get_n_elements_dict(20, True, rankings_sorted)
    return rankings


def compute_page_rank_improved(graph, a, iter_number):
    rankings = page_rank_improved(graph, a, iter_number)

    for k, v in rankings.items():
        rankings[k] = round(v, 2)

    rankings_sorted = {k: v for k, v in sorted(rankings.items(), key=lambda item: item[1])}

    get_n_elements_dict(20, False, rankings_sorted)
    get_n_elements_dict(20, True, rankings_sorted)

def create_histogram(x, y):

    max_value = max(x)
    min_value = min(x)

    bins = np.logspace(0, max_value - min_value, 50)
    plt.hist(y, bins=bins)
    plt.xlabel('Pagerank')
    plt.ylabel('Count')
    plt.show()

def compute_histogram(rankings):
    df = pd.DataFrame(rankings.items(), columns=['Node', 'Score'])
    df_grouped = df.groupby('Score')['Node'].count()
    create_histogram(df_grouped.values, df_grouped.axes[0].values)

start_time = time.time()

data = defaultdict(list)

with open('../data/preprocessed_web-Google.csv', 'rt')as f:
    array = f.readlines()
    for row in array:
        if row == "\n":
            continue
        key, value = row.split(',')[0].strip(), row.split(',')[1].strip()
        data[key].append(value)


print(TIME_PRINT % (time.time() - start_time))
print("Graph Initiliazed Time")

start_time = time.time()
rankings = compute_page_rank(data, 10)
compute_histogram(rankings)
print(TIME_PRINT % (time.time() - start_time))
print("Above results for simple pagerank, 10 iterations")

start_time = time.time()
compute_page_rank(data, 50)
print(TIME_PRINT % (time.time() - start_time))
print("Above results for simple pagerank, 50 iterations")

start_time = time.time()
compute_page_rank(data, 100)
print(TIME_PRINT % (time.time() - start_time))
print("Above results for simple pagerank, 100 iterations")

start_time = time.time()
compute_page_rank(data, 200)
print(TIME_PRINT % (time.time() - start_time))
print("Above results for simple pagerank, 200 iterations")

start_time = time.time()
compute_page_rank_improved(data, 0.2, 10)
print(TIME_PRINT % (time.time() - start_time))
print("Above results for improved pagerank, a=0,2, 10 iterations")

start_time = time.time()
compute_page_rank_improved(data, 0.85, 10)
print(TIME_PRINT % (time.time() - start_time))
print("Above results for improved pagerank, a=0,85, 10 iterations")

start_time = time.time()
compute_page_rank_improved(data, 0.85, 50)
print(TIME_PRINT % (time.time() - start_time))
print("Above results for improved pagerank, a=0,85, 50 iterations")

start_time = time.time()
compute_page_rank_improved(data, 0.85, 100)
print(TIME_PRINT % (time.time() - start_time))
print("Above results for improved pagerank, a=0,85, 100 iterations")

start_time = time.time()
compute_page_rank_improved(data, 0.85, 200)
print(TIME_PRINT % (time.time() - start_time))
print("Above results for improved pagerank, a=0,85, 200 iterations")

start_time = time.time()
compute_page_rank_improved(data, 0.2, 50)
print(TIME_PRINT % (time.time() - start_time))
print("Above results for improved pagerank, a=0,2, 50 iterations")

start_time = time.time()
compute_page_rank_improved(data, 0.2, 100)
print(TIME_PRINT % (time.time() - start_time))
print("Above results for improved pagerank, a=0,2, 100 iterations")

start_time = time.time()
compute_page_rank_improved(data, 0.2, 200)
print(TIME_PRINT % (time.time() - start_time))
print("Above results for improved pagerank, a=0,2, 200 iterations")
