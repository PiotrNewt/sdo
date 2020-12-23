import numpy as np
import lab_pb2
from network.dqn_agent import Agent
from network.util import prepare_trees
from collections import deque

# worker represents a handler for a query
class Worker(object):
    def __init__(self):
        super(Worker, self).__init__()
        self.numOfPlan = 22
        self.agent = Agent(state_size=self.numOfPlan, action_size=7, seed=0)
        self.scores = []
        self.scores_window = deque(maxlen=100)
        self.score = 0
        self.eps = 1.0
        self.handleSQL = ""
        self.i_episode = 0

    def printInfo(self, request):
        print("---------\ndone:\t{}\nsql:\t{}\nreward:\t{}\nplan:\t{}\nepis:\t{}\nscore:\t{}\n".format(
            request.done,
            request.sql,
            request.reward,
            request.plan,
            self.i_episode,
            self.score
        ))
        
    # handleReq handle the request from DB(Client).
    # and if it is the first time for sql to request, we set a handleSQL which is the sql form DB.
    # and we clean the random key when sql done.
    def handleReq(self, request):
        print("handling")
        if self.handleSQL != "" and self.handleSQL != request.sql:
            # TODO: if a sql put in busy work, we do not process it.
            return None

        if self.handleSQL == "":
            self.env_reset(request.sql)

        # sync last step reward
        self.score += request.reward

        self.printInfo(request)

        # if done, we just add last reward
        if request.done:
            return self.done()

        # do train
        state = self.deserialize(request.plan)
        action = self.agent.act(state, self.eps)
        return self.env_step(action)

    # done clean the worker, update epsilon, settlement score
    def done(self):
        self.scores_window.append(self.score)
        self.scores.append(self.score)
        self.eps = max(0.01, 0.995*self.eps)
        res = lab_pb2.NextApplyIdxResponse(
            sql = self.handleSQL,
        )
        self.handleSQL = ""
        if self.i_episode % 10 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(self.i_episode, np.mean(self.scores_window)))
        return res

    # respense the rule index request with action(idx)
    def env_step(self, action):
        return lab_pb2.NextApplyIdxResponse(
            sql = self.handleSQL,
            ruleIdx = action
        )

    def env_reset(self, sql):
        self.handleSQL = sql
        self.score = 0
        self.i_episode += 1

    def tree2trees(self, trees):
        def left_child(x):
            assert isinstance(x, tuple)
            if len(x) == 1:
                # leaf.
                return None
            return x[1]

        def right_child(x):
            assert isinstance(x, tuple)
            if len(x) == 1:
                # leaf.
                return None
            return x[2]

        def transformer(x):
            return np.array(x[0])
        
        return prepare_trees(trees, transformer, left_child, right_child)

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
        return self.tree2trees([self.recursiveDeserialize(0, strs)])

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
    def training(self, plan):
        state = self.deserialize(plan)
        action = self.agent.act(state, self.eps)
        next_state, reward, done = self.env_step(action)
        state = next_state
        pass

    # func for using model
    def using(self):
        pass

    # reference (https://github.com/udacity/deep-reinforcement-learning/blob/master/dqn/solution/Deep_Q_Network_Solution.ipynb)