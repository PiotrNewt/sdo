import numpy as np
import lab_pb2
from network.dqn_agent import Agent
from network.util import prepare_trees
from collections import deque

printDebugInfo = True
windowSize = 5
numOfPlan = 22

# worker represents a handler for a query
class Worker(object):
    def __init__(self):
        super(Worker, self).__init__()
        self.agent = Agent(state_size=numOfPlan, action_size=8, seed=0)
        self.state = ()
        self.next_state = ()
        self.action = 0
        self.scores = []
        self.scores_window = deque(maxlen=windowSize)
        self.score = 0
        self.eps = 1.0
        self.handleSQL = ""
        self.i_episode = 0
        self.idxSeq = []
        self.sqlExeLatency = 0
        self.waitFinalLatency = False

    def printInfo(self, request):
        if printDebugInfo:
            print("---------\ndone:\t\t{}\nsql:\t\t{}\nlatency:\t{}\nparsLatency:\t{}\ncompLatency:\t{}\nplan:\t\t{}\ndbstep:\t\t{}\nepis:\t\t{}\nscore:\t\t{}".format(
                request.done,
                request.sql,
                request.latency,
                request.parserLatency,
                request.compileLatency,
                request.plan,
                request.stepIdx,
                self.i_episode,
                self.score
            ))
        
    # latrncy2reward trans the sql execution laterncy to reward.
    def latency2reward(self, request):
        if request.latency is None or request.latency == 0:
            return 0
        if request.latency < 0:
            return request.latency * 10
        realLatency = request.latency - request.compileLatency - request.parserLatency
        late = realLatency/1000000000
        self.sqlExeLatency = late
        # for tpch-q1
        if late > 2:
            return -(late-2)
        return 2/(late-1)

    # handleReq handle the request from DB(Client).
    # and if it is the first time for sql to request, we set a handleSQL which is the sql form DB.
    # and we clean the handleSQL when sql done.
    def handleReq(self, request):
        if printDebugInfo:
            print("handling")

        # if just a done request without pre-process, we drop it.
        # if self.handleSQL != "" and self.handleSQL != request.sql:
        #     return None

        # if a sql put in busy work, we do not process it.
        if self.handleSQL == "" and request.done:
            if printDebugInfo:
                print("handleSQL:{}\nrequestSQL:{}\ndone?:{}".format(self.handleSQL, request.sql, request.done))
            return None

        # handle the fail query!
        if self.waitFinalLatency and request.done:
            reward = self.latency2reward(-2)
            self.agent.step(self.state, self.action-4, reward, self.next_state, True)
            self.score += reward
            _ = self.done()  # clean the fail query

        # process with final plan
        if request.done and request.latency == 0.0:
            if printDebugInfo:
                print("---------\n*handle final plan*\n")
            self.next_state = self.request2state(request)
            # use to mark setting the fail query reward and reset the v-env.
            self.waitFinalLatency = True
            return None

        # if it is the first call, we reset the env
        if self.handleSQL == "":
            self.env_reset(request.sql)
            self.state = self.request2state(request)
        else:
            # sync last step reward
            reward = self.latency2reward(request)
            if request.plan is not None and request.plan != "":
                self.next_state = self.request2state(request)
            self.agent.step(self.state, self.action-4, reward, self.next_state, request.done)
            self.state = self.next_state
            self.score += reward

        self.printInfo(request)

        # if done, we just add last reward
        if request.done:
            return self.done()

        # get action, in tidb, rule idx will start at 4, so we add 4 and cut down it when model step.
        self.action = self.agent.act(self.state, self.eps) + 4
        return self.env_step(self.action)

    # done clean the worker, update epsilon, settlement score
    def done(self):
        self.scores_window.append(self.score)
        self.scores.append(self.score)
        self.eps = max(0.01, 0.995*self.eps)
        res = lab_pb2.NextApplyIdxResponse(
            sql = self.handleSQL,
        )
        if self.i_episode % windowSize == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}\tLatency: {}\tIdxSeq: {}'.format(self.i_episode, np.mean(self.scores_window), self.sqlExeLatency, self.idxSeq))
        if printDebugInfo:
            print("*idxSeq*:\t{}\n".format(self.idxSeq))
        self.idxSeq = []
        self.handleSQL = ""
        self.waitFinalLatency = False
        return res

    # respense the rule index request with action(idx)
    def env_step(self, action):
        if printDebugInfo:
            print("*idx*:\t\t{}\n".format(action))
        self.idxSeq.append(action)
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
        if i >= len(strs) or strs[i] == "#" or strs[i] == "":
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

    def onehot(self, v, n):
        vec = np.zeros(n)
        vec[v] = 1
        return tuple(vec.tolist())

    def uint2onehot(self, v, n):
        vec = np.zeros(n)
        for i in range(0,n):
            if v & (1 << i) != 0:
                vec[i] = 1
        return tuple(vec.tolist())

    # func for feature to vector(or matrix)
    def feature2vec(self, value, encode="onehot", n = numOfPlan):
        if encode == "onehot":
            return self.onehot(value, n)
        if encode == "u2onehot":
            return self.uint2onehot(value, n)
        return

    # request2state trans request to state
    def request2state(self, request):
        stepIdxEncode = self.feature2vec(request.stepIdx, encode="onehot", n=8)
        flagEncode = self.feature2vec(request.flag, encode="u2onehot", n=14)
        step_flagEncode = (stepIdxEncode+flagEncode,)
        planEncode = self.deserialize(request.plan)
        return self.tree2trees([planEncode, step_flagEncode])

    # func for using model
    def using(self):
        pass

    # reference (https://github.com/udacity/deep-reinforcement-learning/blob/master/dqn/solution/Deep_Q_Network_Solution.ipynb)