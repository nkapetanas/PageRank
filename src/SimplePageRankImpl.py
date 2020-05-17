from src.graph import Graph
from src.node import Node
import numpy as np
import matplotlib.pyplot as plt
import time

# WEB_GOOGLE_TEXT_FILE = "../data/web-Google.txt"
#WEB_GOOGLE_TEXT_FILE = "../data/webSample2.txt"
WEB_GOOGLE_TEXT_FILE = "../data/webSample.txt"
TIME_PRINT = "--- %s seconds ---"


def read_file(file):
    text_lines = list()

    with open(file, "r") as f:
        for line in f:
            line = line.partition('#')[0]
            if line.strip():
                text_lines.append(line.rstrip())
    return text_lines


def page_rank(graph, num_iters):
    # initialize a dictionary with node value and a unit of importance
    pages_with_score = {node: 1 for node in graph.get_nodes()}

    for _ in range(num_iters):
        old_pages_with_score = dict(pages_with_score)

        for page, score in pages_with_score.items():
            connections = graph.get_connections_from_specific_node(page)
            new_score = 0
            for connection in connections:
                new_score += old_pages_with_score[connection] / graph.get_number_of_connections_from_specific_node(
                    connection)

            pages_with_score[page] = new_score
    return pages_with_score


def page_rank_imporved(graph, a, num_iters):
    # initialize a dictionary with node value and a unit of importance
    pages_with_score = {node: 1 for node in graph.get_nodes()}

    norm_part = (1 - a) * np.ones(len(graph.get_nodes()))

    for _ in range(num_iters):
        old_pages_with_score = dict(pages_with_score)

        for page, score in pages_with_score.items():
            connections = graph.get_connections_from_specific_node(page)
            new_score = 0
            for connection in connections:
                new_score += old_pages_with_score[connection] / graph.get_number_of_connections_from_specific_node(
                    connection)
            new_score = new_score * a + norm_part[0]
            pages_with_score[page] = new_score
    return pages_with_score


def initiliaze_graph(text):
    graph = Graph()

    for line in text:
        graph.add_node(Node(line.partition('\t')[0]))
    for line in text:
        graph.add_edge(line.partition('\t')[0], (line.partition('\t')[2]))

    return graph


def get_n_elements_dict(n, reversed_list, sorted_dict):
    if reversed_list:
        for key in list(reversed(list(sorted_dict)))[0:3]:
            print("key {}, value {} ".format(key, sorted_dict[key]))
        return True

    for key in list(sorted_dict)[0:n]:
        print("key {}, value {} ".format(key, sorted_dict[key]))


def compute_page_rank(a, iter_number):
    text = read_file(WEB_GOOGLE_TEXT_FILE)

    graph = initiliaze_graph(text)

    rankings = page_rank(graph, iter_number)

    rankings_sorted = {k: v for k, v in sorted(rankings.items(), key=lambda item: item[1])}

    get_n_elements_dict(20, False, rankings_sorted)
    get_n_elements_dict(20, True, rankings_sorted)


def compute_page_rank_improved(a, iter_number):
    text = read_file(WEB_GOOGLE_TEXT_FILE)

    graph = initiliaze_graph(text)

    rankings = page_rank_imporved(graph, a, iter_number)

    rankings_sorted = {k: v for k, v in sorted(rankings.items(), key=lambda item: item[1])}

    get_n_elements_dict(20, False, rankings_sorted)
    get_n_elements_dict(20, True, rankings_sorted)


def create_histogram(x, y):
    plt.bar(x, y, align='center')
    plt.xlabel('Pagerank')
    plt.ylabel('Count')
    plt.show()


start_time = time.time()
compute_page_rank(0.2, 50)
print(TIME_PRINT % (time.time() - start_time))

start_time = time.time()
compute_page_rank(0.2, 100)
print(TIME_PRINT % (time.time() - start_time))

start_time = time.time()
compute_page_rank(0.2, 200)
print(TIME_PRINT % (time.time() - start_time))

start_time = time.time()
compute_page_rank_improved(0.2, 50)
print(TIME_PRINT % (time.time() - start_time))

start_time = time.time()
compute_page_rank_improved(0.2, 100)
print(TIME_PRINT % (time.time() - start_time))

start_time = time.time()
compute_page_rank_improved(0.2, 200)
print(TIME_PRINT % (time.time() - start_time))
