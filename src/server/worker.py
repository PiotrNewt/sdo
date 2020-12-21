import numpy as np

# worker represents a handler for a query
class Worker(object):
    def __init__(self, parameter):
        super(Worker, self).__init__()
        self.praram = parameter
        self.numOfPlan = 22
        # TODO: add parameter which from request

    # recursiveDeserialize recursive make strs to tree
    def recursiveDeserialize(self, i, strs):
        if i >= len(strs) or strs[i] == "#":
            return None

        node = (self.feature2vec(int(strs[i])),)
        # idx * 2 + 1 left
        left = self.recursiveDeserialize(i*2+1, strs)
        if left is not None:
            node += (left,)
        # idx * 2 + 2 right
        right = self.recursiveDeserialize(i*2+2, strs)
        if right is not None:
            node += (right,)
        elif len(node) == 2:    # tcnn need all nodes must have both a left and a right child or no children
            node += ((self.feature2vec(21),),)
        return node

    # deserialize deserializes tidb(client) encoded data(string) to tree.
    def deserialize(self, data):
        if data == "":
            return None
        strs = str.split(data, "_")
        return self.recursiveDeserialize(0, strs)

    def onehot(self, v):
        vec = np.zeros(self.numOfPlan)
        vec[v] = 1
        return tuple(vec.tolist())

    # func for feature to vector(or matrix)
    def feature2vec(self, value, encode="onehot"):
        if encode == "onehot":
            return self.onehot(value)
        return

    # func for training (maybe we need to set checkpoint for using)
    def training(self):
        pass

    # func for using
    def using(self):
        pass

    # reference (https://github.com/udacity/deep-reinforcement-learning/blob/master/dqn/solution/Deep_Q_Network_Solution.ipynb)