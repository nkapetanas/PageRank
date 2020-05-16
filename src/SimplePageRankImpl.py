from src.graph import Graph
from src.node import Node

# WEB_GOOGLE_TEXT_FILE = "../data/web-Google.txt"
WEB_GOOGLE_TEXT_FILE = "../data/webSample2.txt"


def read_file(file):
    text_lines = list()

    with open(file, "r") as f:
        for line in f:
            line = line.partition('#')[0]
            if line.strip():
                text_lines.append(line.rstrip())
    return text_lines


def page_rank(graph, num_iters=10):
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


def initiliaze_graph(text):
    graph = Graph()

    for line in text:
        graph.add_node(Node(line.partition('\t')[0]))
    for line in text:
        graph.add_edge(line.partition('\t')[0], (line.partition('\t')[2]))

    return graph


text = read_file(WEB_GOOGLE_TEXT_FILE)
graph = initiliaze_graph(text)

page_rank(1, graph)
