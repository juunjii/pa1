import sys
import os
import threading
import random
import glob
from collections import deque
import numpy as np


sys.path.append('gen-py')
sys.path.insert(0, glob.glob('../thrift-0.19.0/lib/py/build/lib*')[0])
from thrift import Thrift
from thrift.transport import TSocket
from thrift.transport import TTransport
from thrift.protocol import TBinaryProtocol
from thrift.server import TServer



# from coordinator import coordinator
from coordinator.ttypes import WeightMatrices
from compute import compute
from ML import *
# from compute.ttypes import WeightMatrices as ComputeWeightMatrices

from ML import mlp, scale_matricies, sum_matricies, calc_gradient

class CoordinatorHandler:
    def __init__(self, scheduling_policy):
        self.scheduling_policy = scheduling_policy
        self.compute_nodes = [] # stores (host, port) tuples
        self.parse_compute_nodes()

    '''
    Parse list of compute nodes from text file; 
    Gets compute nodes' respective ip and port
    '''
    def parse_compute_nodes(self):
        try:
            with open('compute_nodes.txt', 'r') as file:
                for line in file:
                    host, port = line.strip().split(',')
                    self.compute_nodes.append((host, int(port)))
        except Exception as e:
            print(f"Error parsing compute nodes: {e}")
            sys.exit(1)
    
    '''
    Coordinator node acts as a client to compute node,
    and attempts to connect to compute nodes
    '''
    def connet_compute_node_server(self, ip, port):
        try:
            transport = TSocket.TSocket(ip, port)
            transport = TTransport.TBufferedTransport(transport)
            protocol = TBinaryProtocol.TBinaryProtocol(transport)
            # Allows remote procedures on compute node server
            client = compute.Client(protocol) 
            transport.open()

        except Exception as e:
            print(f"Error: {e}")
            if 'transport' in locals():
                transport.close()
            sys.exit(1)

        return client, transport
    
    def populate_queue(self, dir):
        # Sanitize input
        if not dir or not os.path.exists(dir) or not os.path.isdir(dir):
            print(f"Error: Invalid directory '{dir}'")
            return None 
       
        work_queue = deque()

        # Loop through files in the 'letters' directory
        for file in os.listdir(dir):
            if file.startswith("train"):
                filename = os.path.join(dir, file)
                work_queue.append(filename)

        return work_queue

    '''
    Handles work scheduling on compute nodes following policies: 
    '1' - Random Scheduling: Injecting Load (delay before task execution)
    '2' - Load-Balancing: Injecting load & rejects task 
    (If compute node rejects job, another compute node gets the job)

    Returns: (host, ip) of available compute node
    '''
    def work_scheduling(self):
        # Random scheduling - Random node gets job
        if self.scheduling_policy == 1:
            return random.choice(self.compute_nodes)
        else:
            # Load Balancing 
            for node in self.compute_nodes:
                # Connect to compute node
                try:
                    # Unpack tuple 
                    transport = TSocket.TSocket(node[0], node[1])
                    transport = TTransport.TBufferedTransport(transport)
                    protocol = TBinaryProtocol.TBinaryProtocol(transport)
                    client = compute.Client(protocol)
                    transport.open()
                    
                    # If compute node rejects job, another compute node gets the job
                    if client.rejectTask():
                        transport.close()
                        return random.choice(self.compute_nodes)
                    else:
                        transport.close()
                        return node
                    
                except Exception as e:
                    print(f"Error: {e}")
                    if 'transport' in locals():
                        transport.close()
                    sys.exit(1)

  

        
    '''
    Runs a series of training rounds, where the entire training dataset is trained in 
    batches on separate compute nodes, with the gradient from each batch averaged and updated 
    on the coordinator’s model. 
    
    Parameters:
    dir: location of the /letters directory
    rounds: training rounds 
    epochs: number of complete pass through the entire dataset
    k: number of outcome units (26 alphabets)
    h: number of hidden units (between 16 - 26)
    eta: learning rate of model (0.0001)
    '''
    def train(self, dir, rounds, epochs, h, k, eta):
        # Connect to compute node
        # client, transport = self.connet_compute_node_server()

        work_queue = self.populate_queue(dir)
                
        # Randomly choose training set to initialize model
        random_training_set = random.choice(work_queue)
        
        model = mlp()

        # Initalize ML model with random weights of dimensions k and h
        success = model.init_training_random(random_training_set, k, h)
        if (success == False):
            raise Exception(f"Model initialization failed with file {random_training_set}")
        
        def worker_thread():
            try:
                training_data = work_queue.pop()

                # Get ip, port of available nodes
                ip, port = self.work_scheduling()
                # Attempt compute node connection
                client, transport = self.connet_compute_node_server(ip, port)

                # Package model weights
                weights = WeightMatrices(V = model.V.tolist(), W = model.W.tolist())

                gradient = client.trainMLP(weights, training_data, eta, epochs)
                with mutex:
                    shared_gradient_V += np.array(gradient.V)
                    shared_gradient_W += np.array(gradient.W)
            except:
                pass

        for r in range(rounds):
            # Added extra rows to account for bias weights
            shared_gradient_V = np.zeros((h + 1,k))
            shared_gradient_W = np.zeros((k + 1, h))
            # create a lock for accessing the shared gradient vars
            mutex = threading.Lock()

            threads = []
            num_nodes = len(self.compute_nodes)

            # get the nodes going through the queue
            for i in range(num_nodes):
                thread = threading.Thread(target=worker_thread)
                threads.append(thread)
                thread.start()

            # wait for threads to work through the queue
            for thread in threads:
                thread.join()
            
            shared_gradient_V = shared_gradient_V / num_nodes
            shared_gradient_W = shared_gradient_W / num_nodes
            model.update_weights(shared_gradient_V, shared_gradient_W)
        return model.validate()



        

if __name__ == '__main__':
    obj = CoordinatorHandler(1)


    print(obj.train("letters", 25, 15, 20, 26, .0001))
    print(obj.train("letters", 15, 50, 24, 26, .0001))