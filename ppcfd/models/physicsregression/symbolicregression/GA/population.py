import numpy as np


class Individual:
    def __init__(self, node, xy, loss=None, complexity=None, len=None):
        self.node = node
        self.xy = xy
        if loss is None:
            self.update()
        else:
            self.loss = loss
            self.complexity = complexity
            self.len = len

    def __len__(self):
        return self.len

    def update(self):
        self._update_loss()
        self._update_len()
        self._update_complexity()

    def _update_loss(self):
        x, y = self.xy
        y_pred = self.node.val(x)
        if any(np.isnan(y_pred)):
            self.loss = np.inf
        try:
            self.loss = np.sqrt(np.mean(np.square(y_pred.reshape(-1) - y.reshape(-1))))
        except:
            self.loss = np.inf
        self.loss = np.mean(np.square(y_pred.reshape(-1) - y.reshape(-1)))

    def _update_len(self):
        self.len = len(self.node)

    def _update_complexity(self):
        self.complexity = len(self)

    def copy(self):
        return Individual(
            self.node.copy(), self.xy, self.loss, self.complexity, self.len
        )

    def __repr___(self):
        return str(self.node)

    def __str__(self):
        return str(self.node)


class Population:
    def __init__(self, population):
        self.population = population

    def replace_oldest(self, individual):
        self.population.pop(0)
        self.population.append(individual)

    def pop_and_add(self, idx, individual):
        self.population.pop(idx)
        self.population.append(individual)

    def __len__(self):
        return len(self.population)

    def __iter__(self):
        self.idx = 0
        return self

    def __next__(self):
        if self.idx < len(self.population):
            result = self.population[self.idx]
            self.idx += 1
            return result
        else:
            raise StopIteration


class BestFrontier:
    def __init__(self):
        self.frontier = {}

    def add(self, individual, complexity):
        if complexity not in self.frontier and not np.isnan(individual.loss):
            self.frontier[complexity] = individual.copy()
        elif (
            complexity in self.frontier
            and individual.loss < self.frontier[complexity].loss
        ):
            self.frontier[complexity] = individual.copy()

    def union(self, frontier):
        assert isinstance(frontier, BestFrontier)
        for complexity, individual in frontier.frontier.items():
            self.add(individual, complexity)

    def random_choose(self):
        idx = np.random.choice(list(self.frontier.keys()))
        return self.frontier[idx]

    def __len__(self):
        return len(self.frontier)

    def __repr__(self):
        dic = sorted(self.frontier.items(), key=lambda x: x[0])
        for k, v in dic:
            print("complexity:{} --> ".format(k), end="")
            print(v, end="")
            print(" --> loss={}".format(v.loss))
        return ""

    def best(self):
        """Return the best node across all complexity"""
        for k, v in self.frontier.items():
            if abs(v.loss) < 1e-06:
                v.loss = 0
        dic = sorted(self.frontier.items(), key=lambda x: (x[1].loss, x[0]))
        return dic[0][1].node
