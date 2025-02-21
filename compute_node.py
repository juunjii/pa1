from ML import *

import sys
import time
import random
import glob
sys.path.append('gen-py')
sys.path.insert(0, glob.glob('../../thrift-0.19.0/lib/py/build/lib*')[0])

from thrift.transport import TSocket
from thrift.transport import TTransport
from thrift.protocol import TBinaryProtocol
from thrift.server import TServer

from compute_node import compute
from compute_node.ttypes import WeightMatrices

class ComputeNodeHandler:

    def __init__(self):
        pass
    def __init__(self, load_probability):
        self.load_probability = float(load_probability)

    def trainMLP(self, weights, data, eta, epochs):
        model = mlp()

        initial_V, initial_W = weights.V, weights.W

        initialized_model = model.init_training_model(data, initial_V, initial_W)
        if (initialized_model == False):
            raise Exception(f"Model initialization failed with file {data}")
        
        error_rate = model.train(eta, epochs)
        if (error_rate == -1):
            raise Exception("Model training failed!")
        
        trained_V, trained_W = model.get_weights()

        gradient_V = calc_gradient(trained_V)
        gradient_W = calc_gradient(trained_W)

        return WeightMatrices(V=gradient_V.tolist(), W=gradient_W.tolist())


    def loadInjection(self):
        if random.random() < self.load_probability:
            time.sleep(3)  # 3-second delay
    
    def rejectTask(self):
        return random.random() < self.load_probability


if __name__ == '__main__':
    handler = ComputeNodeHandler()
    processor = compute.Processor(handler)
    transport = TSocket.TServerSocket(host='127.0.0.1', port=9091)
    tfactory = TTransport.TBufferedTransportFactory()
    pfactory = TBinaryProtocol.TBinaryProtocolFactory()

    server = TServer.TSimpleServer(processor, transport, tfactory, pfactory)

    # You could do one of these for a multithreaded server
    # server = TServer.TThreadedServer(
    #     processor, transport, tfactory, pfactory)
    # server = TServer.TThreadPoolServer(
    #     processor, transport, tfactory, pfactory)

    print('Starting the server...')
    server.serve()
    print('done.')