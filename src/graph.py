from src.node import Node


class Graph(object):
    nodes = dict()

    def add_node(self, node):
        if isinstance(node, Node) and node.key not in self.nodes:
            self.nodes[node.key] = node
            return True

        return False

    def add_edge(self, node1, node2):
        if node1 in self.nodes and node2 in self.nodes:
            for key, value in self.nodes.items():
                if key == node1:
                    value.add_neighbor(node2)
                if key == node2:
                    value.add_neighbor(node1)
            return True
        return False

    def get_nodes(self):
        return self.nodes.keys()

    def print_graph(self):
        for key in sorted(list(self.nodes.keys())):
            print(key + str(self.nodes[key].connected_to))


g = Graph()
a = Node('A')
g.add_node(a)
g.add_node(Node('B'))

for i in range(ord('A'), ord('K')):
    g.add_node(Node(chr(i)))

edges = ['AB', 'AE', 'BF', 'CG', 'DE', 'DH', 'EH', 'FG', 'FI', 'FJ', 'GJ', 'HI']

for edge in edges:
    g.add_edge(edge[:1], edge[1:])

g.print_graph()
