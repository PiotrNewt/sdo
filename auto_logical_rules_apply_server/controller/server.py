import grpc
import lab_pb2
import lab_pb2_grpc
import time
import multiprocessing

from concurrent import futures
from worker import worker

# config
port = "localhost:50051"

class service(lab_pb2_grpc.AutoLogicalRulesApplyServicer):
    # init
    def __init__(self):
        self.worker = worker()

    # imp rpc func
    def getNextApplyIdxRequest(self, request, context):
        if self.worker is None:
            self.worker = worker()

        # TODO: code for training.

        return lab_pb2.NextApplyIdxResponse(
            sqlFingerPrint = request.sqlFingerPrint,
            sql = request.sql
        )


# thread pool = num of cpu cores
server = grpc.server(futures.ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()))
lab_pb2_grpc.add_LogicalOptmizerApplyOrderServicer_to_server(service(), server)

# set service port
server.add_insecure_port(port)
server.start()

# listening
try:
    while True:
        time.sleep(60*60*24)
except KeyboardInterrupt:
    server.stop(0)