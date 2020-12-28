import lab_pb2
from worker import Worker

worker_test = Worker()

def actionTest():
    for i in range(0,8):
        action = worker_test.handleReq(lab_pb2.NextApplyIdxRequest(
            sql = "select sum(t.c) from t where a < 1 group by s;",
            latency = 0,
            done = False,
            plan = "19_1_#_#_#",
            flag = 4132,
            stepIdx = i,
        ))
        print("{} : {}".format(type(action), action))

def encodeTest():
    state = worker_test.request2state(lab_pb2.NextApplyIdxRequest(
        sql = "select sum(t.c) from t where a < 1 group by s;",
        latency = 0,
        done = False,
        plan = "19_1_#_#_#",
        flag = 4132,
        stepIdx = 7,
    ))
    print(state)

# encodeTest()
actionTest()