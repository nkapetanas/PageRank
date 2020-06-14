from collections import defaultdict
import pandas as pd

WEB_GOOGLE_TEXT_FILE = "../data/web-Google.txt"
WEB_GOOGLE_TEXT_FILE_CSV_WITH_HEADER = "../data/web-Google2_with_header.txt"
TIME_PRINT = "--- %s seconds ---"


def read_csv_file(file_path):
    return pd.read_csv(file_path, encoding="utf-8", skip_blank_lines=True)


def write_df_to_csv(df):
    with open('preprocessed_web-Google.csv', "w", encoding="utf-8") as file:
        df.to_csv(file, mode="a", header=False, index=False)


def read_file(file):
    text_lines = list()

    with open(file, "r") as f:
        for line in f:
            line = line.partition('#')[0]
            if line.strip():
                text_lines.append(line.rstrip())
    return text_lines


def preprocess(graph, df):
    pages_with_score = {node: 1 for node in graph.keys()}
    temp_pages_with_score = dict(pages_with_score)

    all_graph_keys = list(graph.keys())
    keys_without_out_values = list()
    for page, score in temp_pages_with_score.items():
        connections = graph[page]

        for connection in connections:
            outgoing_edges_from_node = len(graph[connection])
            if outgoing_edges_from_node == 0:
                keys_without_out_values.append(connection)
                pages_with_score[connection] = 1
                graph[connection] = all_graph_keys

    for value in keys_without_out_values:
        df = df[df.ToNodeId != int(value)]

    write_df_to_csv(df)


df = read_csv_file("../data/preprocessed_web-Google-Header.csv")
data = defaultdict(list)

with open('../data/preprocessed_web-Google.csv', 'rt')as f:
    array = f.readlines()
    for row in array:
        if row == "\n":
            continue
        key, value = row.split(',')[0].strip(), row.split(',')[1].strip()
        data[key].append(value)

preprocess(data, df)
