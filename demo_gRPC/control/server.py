import grpc
import order_pb2
import order_pb2_grpc
import time
import multiprocessing

from concurrent import futures

class service(order_pb2_grpc.LogicalOptmizerApplyOrderServicer):
    # imp rpc func
    def getApplyOrderRequest(self, request, context):
        return order_pb2.ApplyOrderResponse(
            sql = request.sql,
            applyOrder = "1234567"
        )

# thread pool = num of cpu cores
server = grpc.server(futures.ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()))
order_pb2_grpc.add_LogicalOptmizerApplyOrderServicer_to_server(service(), server)

# set service port
server.add_insecure_port("localhost:50051")
server.start()

# listening
try:
    while True:
        time.sleep(60*60*24)
except KeyboardInterrupt:
    server.stop(0)