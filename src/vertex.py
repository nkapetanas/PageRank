class Vertex:
    def __init__(self, key):
        self.key = key
        self.connected_to = list()

    def add_neighbor(self, neighbor):
        if neighbor not in self.connected_to:
            self.connected_to.append(neighbor)
            self.connected_to.sort()

    def __str__(self):
        return str(self.key) + " is connected to: " + str([element for element in self.connected_to])

    def get_connections(self):
        return self.connected_to

    def get_key(self):
        return self.key
