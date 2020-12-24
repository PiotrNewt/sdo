import lab_pb2
from worker import Worker

worker_test = Worker()

action = worker_test.handlerReq(lab_pb2.NextApplyIdxRequest(
    sqlFingerPrint = "xcdfefs",
    sql = "select sum(t.c) from t where a < 1 group by s;",
    latency = 0,
    done = False,
    plan = "19_1_#_#_#",
    flag = "",
))
print("{} : {}".format(type(action), action))