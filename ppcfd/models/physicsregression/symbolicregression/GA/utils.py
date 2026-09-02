import numpy as np


def bfgs_gp(node, xy, oracle, include_const=True):
    """
    const optimize for genetic programming

    first take out all the const in the given node,

    then do the optimize steps,

    and finally replace the const back into the node.

    if there are no or more than 6 const, then we refuse to do optimize

    Parameters:
    -----------
    node : Node
        the node form of expression to optimize

    xy : tuple
        input-output data pairs, of shape ((n_data, n_features), (n_data, 1))

    Returns:
    --------
    node : Node
        the node form of expression after optimize
    """
    safe_types = ["id", "safe", "neg", "inv", "safe-neg", "safe-inv"]
    if (
        include_const
        and len(node) < node.params.max_complexity - 5
        and np.random.random() < 0.6
    ):
        safe_types.append("linear")
        use_const = True
    else:
        use_const = False
    refined_node = oracle.safely_refine(xy[0], xy[1], node, "id", safe_types)
    if use_const and len(refined_node) == len(node) + 5:
        if abs(float(node.children[0].children[1].value) - 1) < 1e-06:
            node.children[0] = node.children[0].children[0]
        if abs(float(node.children[1].value)) < 1e-06:
            node = node.children[0]
    return refined_node


class TimeRecorder:
    def __init__(self):
        self.timerecorder1 = {"evolve": 0, "simplify": 0, "optimize": 0, "migration": 0}
        self.timerecorder2 = {}

    def record(self, item, time):
        if item in self.timerecorder1:
            self.timerecorder1[item] += time
        elif item in self.timerecorder2:
            self.timerecorder2[item] += time
        else:
            self.timerecorder2[item] = time

    def write_in(self, path):
        s = "time recorder:\n\n"
        for k, v in self.timerecorder1.items():
            s += "{}:{:4f}\n".format(k, v)
        s += "\n"
        for k, v in self.timerecorder2.items():
            s += "{}:{:4f}\n".format(k, v)
        with open(path, "w") as fi:
            fi.write(s)


class DataRecorder:
    def __init__(self):
        self.datarecorder = []

    def new_epoch(self):
        self.datarecorder.append([])

    def new_population(self, population):
        self.datarecorder[-1].append(
            {
                "populations": {
                    "original": [
                        (str(individual.node), individual.loss)
                        for individual in population
                    ]
                },
                "mutations": [],
                "simplifys": [],
                "optimizes": [],
            }
        )

    def record_population(self, population, name):
        self.datarecorder[-1][-1]["populations"][name] = [
            (str(individual.node), individual.loss) for individual in population
        ]

    def record_mutation(self, mutate_op, old, new):
        self.datarecorder[-1][-1]["mutations"].append(
            [mutate_op, str(old.node), old.loss, str(new.node), new.loss]
        )

    def record_cross(self, old1, old2, new1, new2):
        self.datarecorder[-1][-1]["mutations"].append(
            [
                "crossover",
                str(old1),
                old1.loss,
                str(old2),
                old2.loss,
                str(new1),
                new1.loss,
                str(new2),
                new2.loss,
            ]
        )

    def record_simplify(self, individual, state):
        if state == "before":
            self.info = str(individual.node)
        elif state == "after":
            self.datarecorder[-1][-1]["simplifys"].append(
                [self.info, str(individual.node)]
            )
        else:
            print("unknown state:{}".format(state))

    def record_optimize(self, individual, state):
        if state == "before":
            self.info = [str(individual.node), individual.loss]
        elif state == "after":
            self.datarecorder[-1][-1]["optimizes"].append(
                [self.info[0], self.info[1], str(individual.node), individual.loss]
            )
        else:
            print("unknown state:{}".format(state))

    def write_in(self, path):
        i = len(self.datarecorder) - 1
        j = len(self.datarecorder[-1]) - 1
        if ".txt" in path:
            path = path[:-4] + f"{i}_{j}.txt"
        else:
            path = path + f"{i}_{j}.txt"
        s = f"epoch:{i}, population:{j}\n\n"
        for name, population in self.datarecorder[i][j]["populations"].items():
            s += f"\n{name}:\n\n"
            for node, loss in population:
                s += "node:{}, loss:{:.4f}\n".format(node, loss)
        s += "\nmutation operation:\n\n"
        for k in self.datarecorder[i][j]["mutations"]:
            if k[0] == "crossover":
                s += """crossr, node1:{}
        node2:{}
         --> {}
         --> {}
          loss:{:.4f},{:.4f} --> {:.4f},{:.4f}

""".format(
                    k[1], k[3], k[5], k[7], k[2], k[4], k[6], k[8]
                )
            else:
                s += "{}, node:{}\n         --> {}\n        loss:{:.4f} --> {:.4f}\n\n".format(
                    k[0], k[1], k[3], k[2], k[4]
                )
        s += "\nsimplify:\n\n"
        for k in self.datarecorder[i][j]["simplifys"]:
            s += f"simplify, node:{k[0]}\n           --> {k[1]}\n\n"
        s += "\noptimize:\n\n"
        for k in self.datarecorder[i][j]["optimizes"]:
            s += "optimize, node:{}\n           --> {}\n          loss:{:.4f} --> {:.4f}\n\n".format(
                k[0], k[2], k[1], k[3]
            )
        with open(path, "w") as fi:
            fi.write(s)
