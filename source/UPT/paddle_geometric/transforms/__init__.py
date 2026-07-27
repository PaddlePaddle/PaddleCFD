from .._compat import knn_graph


class KNNGraph:
    def __init__(self, k, loop=False, force_undirected=False, flow='source_to_target'):
        self.k = k
        self.loop = loop
        self.force_undirected = force_undirected
        self.flow = flow

    def __call__(self, data):
        batch = getattr(data, 'batch', None)
        data.edge_index = knn_graph(
            x=data.pos,
            k=self.k,
            batch=batch,
            loop=self.loop,
            flow=self.flow,
            force_undirected=self.force_undirected,
        )
        return data
