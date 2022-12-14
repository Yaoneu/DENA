"""Graph utilities."""
import networkx as nx

class Graph(object):
    def __init__(self):
        self.G = None
        self.look_up_dict = {}
        self.look_back_list = []
        self.node_size = 0

    def encode_node(self):
        look_up = self.look_up_dict
        look_back = self.look_back_list
        for node in self.G.nodes():
            look_up[node] = self.node_size
            look_back.append(node)
            self.node_size += 1
            self.G.nodes[node]['status'] = ''

    def read_g(self, g):
        self.G = g
        self.encode_node()

    def add_edges(self,edge_list):
        self.G = nx.DiGraph()
        for el in edge_list:
            self.G.add_edge(str(el[0]), str(el[1]))

    def read_edgelist(self, edgelist, weighted=False):
        self.G = nx.DiGraph()

        def read_unweighted(el):
            src = el[0]
            dst = el[1]
            self.G.add_edge(src, dst)
            self.G[src][dst]['weight'] = 1.0

        def read_weighted(el):
            src = el[0]
            dst = el[1]
            w = el[2]
            self.G.add_edge(src, dst)
            self.G[src][dst]['weight'] = float(w)

        func = read_unweighted
        if weighted:
            func = read_weighted
        for el in edgelist:
            func(el)

        self.encode_node()

    def read_node_status(self, filename):
        fin = open(filename, 'r')
        while 1:
            l = fin.readline()
            if l == '':
                break
            vec = l.split()
            self.G.nodes[vec[0]]['status'] = vec[1]  # train test valid
        fin.close()
