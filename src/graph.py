from src.vertex import Vertex


class Graph(object):
    vertices = dict()

    def add_vertex(self, vertex):
        if isinstance(vertex, Vertex) and vertex.key not in self.vertices:
            self.vertices[vertex.key] = vertex
            return True

        return False

    def add_edge(self, edge1, edge2):
        pass # TODO
