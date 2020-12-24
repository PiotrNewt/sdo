import grpc
# import lab_pb2
import lab_pb2_grpc
import time
import multiprocessing

from concurrent import futures
from worker import Worker

# config
port = "127.0.0.1:50051"

class service(lab_pb2_grpc.AutoLogicalRulesApplyServicer):
    # init
    def __init__(self):
        print("server generating...")
        self.worker = Worker()
        print("done! port:[{}]\nlistening...".format(port))

    # imp rpc func
    def getNextApplyIdxRequest(self, request, context):
        if self.worker is None:
            self.worker = Worker()

        response = self.worker.handleReq(request)
        return response


# thread pool = num of cpu cores
server = grpc.server(futures.ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()))
lab_pb2_grpc.add_AutoLogicalRulesApplyServicer_to_server(service(), server)

# set service port
server.add_insecure_port(port)
server.start()

# listening
try:
    while True:
        time.sleep(60*60*24)
except KeyboardInterrupt:
    server.stop(0)