

import sys
import time
import random
import glob
import os 

import numpy as np
sys.path.append('gen-py')
sys.path.insert(0, glob.glob('../thrift-0.19.0/lib/py/build/lib*')[0])

from thrift.transport import TSocket
from thrift.transport import TTransport
from thrift.protocol import TBinaryProtocol
from thrift.server import TServer

# from compute_node import compute
from compute.ttypes import WeightMatrices

from ML import *

class ComputeNodeHandler:

    # def __init__(self, load_probability):
    #     self.load_probability = float(load_probability)

    def trainMLP(self, weights, data, eta, epochs):
        model = mlp()

        # Unpacking tuple
        initial_V, initial_W = weights

        # Initialize model before training
        initialized_model = model.init_training_model(data, initial_V, initial_W)
        if (initialized_model == False):
            raise Exception(f"Model initialization failed with file {data}")

        # Train
        training_error_rate = model.train(eta, epochs)
        if (training_error_rate == -1):
            raise Exception("Model training failed!")
        print(f"-----Training Error Rate: {training_error_rate}")

        trained_V, trained_W = model.get_weights()

        # Calculate gradient from weights post model training
        gradient_V = calc_gradient(trained_V, initial_V) 
        gradient_W = calc_gradient(trained_W, initial_W) 
        
        # Validating
        validation_file = "letters/validate_letters.txt"
        error_rate = model.validate(validation_file)
        print(f"-----Validation Error Rate: {error_rate}")
           
        return WeightMatrices(V=gradient_V.tolist(), W=gradient_W.tolist())

    def loadInjection(self):
        if random.random() < self.load_probability:
            time.sleep(3)  # 3-second delay
    
    def rejectTask(self):
        return random.random() < self.load_probability


if __name__ == '__main__':

    obj = ComputeNodeHandler()

    mlp_test = mlp()

    file = "letters/train_letters1.txt"
    
    mlp.init_training_random(mlp_test, file, 26, 20)

    weights = mlp_test.get_weights()

    trained_model = obj.trainMLP(weights, file, 0.001, 15)

   


    # handler = ComputeNodeHandler()
    # processor = compute.Processor(handler)
    # transport = TSocket.TServerSocket(host='127.0.0.1', port=9091)
    # tfactory = TTransport.TBufferedTransportFactory()
    # pfactory = TBinaryProtocol.TBinaryProtocolFactory()

    # server = TServer.TSimpleServer(processor, transport, tfactory, pfactory)

    # You could do one of these for a multithreaded server
    # server = TServer.TThreadedServer(
    #     processor, transport, tfactory, pfactory)
    # server = TServer.TThreadPoolServer(
    #     processor, transport, tfactory, pfactory)

    # print('Starting the server...')
    # server.serve()
    # print('done.')