

import sys
import time
import random
import glob
import os 

sys.path.append('gen-py')
sys.path.insert(0, glob.glob('../thrift-0.19.0/lib/py/build/lib*')[0])

from thrift import Thrift
from thrift.transport import TSocket
from thrift.transport import TTransport
from thrift.protocol import TBinaryProtocol
from thrift.server import TServer

from compute import compute
from compute.ttypes import WeightMatrices

from ML import *

class ComputeNodeHandler:

    def __init__(self, load_probability):
        self.load_probability = float(load_probability)

    '''
    Receive weights matrices and a training data filename from the
    coordinator node to perform a round of training. 
    '''
    def trainMLP(self, weights, data, eta, epochs):
        # Delays based on load probability 
        self.loadInjection()

        model = mlp()

        # Unpacking tuple
        initial_V, initial_W = weights.V, weights.W

        # Initialize model before training
        initialized_model = model.init_training_model(data, initial_V, initial_W)
        if (initialized_model == False):
            raise Exception(f"Model initialization failed with file {data}")

        # Train
        training_error_rate = model.train(eta, epochs)
        if (training_error_rate == -1):
            raise Exception("Model training failed!")
        print(f"-----Training Error Rate: {training_error_rate}")

        # New weights
        trained_V, trained_W = model.get_weights()

        # Calculate gradient from weights post model training
        gradient_V = calc_gradient(trained_V, initial_V) 
        gradient_W = calc_gradient(trained_W, initial_W) 
        
        # Validating
        validation_file = "letters/validate_letters.txt"
        error_rate = model.validate(validation_file)
        print(f"-----Validation Error Rate: {error_rate}")
           
        return WeightMatrices(V=gradient_V.tolist(), W=gradient_W.tolist())

    '''
    Simulate delay before executing tasks in the compute node
    based on load probability.
    '''
    def loadInjection(self):
        if random.random() < self.load_probability:
            time.sleep(3)  # 3-second delay
    
    '''
    In load-balancing scheduling, compute  nodes will have a ‘load 
    probability’ chance to reject a task 
    '''
    def rejectTask(self):
        return random.random() < self.load_probability


def main():
    # Read ip and port from command line
    if len(sys.argv) < 3:
        print("python3 ex_client.py <port> <load_probability>")
        sys.exit(1)
    
    # Parse command line arguments
    port = sys.argv[1]
    load_probability = float(sys.argv[2])

    if (load_probability < 0) or (load_probability > 1):
        print("Load probability must be a value between 0 and 1.")
        sys.exit(1)
    
    handler = ComputeNodeHandler(load_probability)
    processor = compute.Processor(handler)
    transport = TSocket.TServerSocket(host='127.0.0.1', port=port)
    tfactory = TTransport.TBufferedTransportFactory()
    pfactory = TBinaryProtocol.TBinaryProtocolFactory()

    server = TServer.TSimpleServer(processor, transport, tfactory, pfactory)

    print('Starting compute node...')
    server.serve()
    print('done.')


if __name__ == '__main__':
    try:
        main()
    except Thrift.TException as tx:
        print('%s' % tx.message)
