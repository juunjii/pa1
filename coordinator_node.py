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
from coordinator import coordinator
from coordinator.ttypes import WeightMatrices
from compute import compute
import random
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
    def connect_compute_node_server(self, ip, port):
        try:
            transport = TSocket.TSocket(ip, port)
            transport = TTransport.TBufferedTransport(transport)
            protocol = TBinaryProtocol.TBinaryProtocol(transport)
            # Allows remote procedures on compute node server
            client = compute.Client(protocol) 
            transport.open()

        except Exception as e:
            print(f"Error in connecting to compute node: {e}")
            # Close connection if exist in local scope 
            if 'transport' in locals():
                transport.close()
            sys.exit(1)

        return client, transport
    
    '''
    Populate the work queue containing training data files 
    for the compute node to train the model
    '''
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
    Populate a list of training data files for initial 
    initialization of the training model
    '''
    def populate_list(self, dir):
        # Sanitize input
        if not dir or not os.path.exists(dir) or not os.path.isdir(dir):
            print(f"Error: Invalid directory '{dir}'")
            return None 
       
        work_list = []

        # Loop through files in the 'letters' directory
        for file in os.listdir(dir):
            if file.startswith("train"):
                filename = os.path.join(dir, file)
                work_list.append(filename)

        return work_list

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
                        print("Compute node rejects")
                        transport.close()
                        return random.choice(self.compute_nodes)
                    else:
                        print("Compute node accepts")
                        transport.close()
                        return node
                    
                except Exception as e:
                    print(f"Error in compute connection during work scheduling: {e}")
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
      
        work_list = self.populate_list(dir)
        # Randomly choose training set to initialize model
        random_training_set = random.choice(work_list)
        
        # Central model
        model = mlp()

        # Initalize ML model with random weights of dimensions k and h
        success = model.init_training_random(random_training_set, k, h)
        if (success == False):
            raise Exception(f"Model initialization failed with file {random_training_set}")
        
        for r in range(rounds):
            print(f"Starting round {r+ 1}/{rounds}")
        
            # create a lock for accessing the shared gradient vars
            mutex = threading.Lock()
            shared_gradient_V = np.zeros((h + 1,k))
            shared_gradient_W = np.zeros((model.W.shape[0], h))
    
            jobs_completed = 0

            work_queue = self.populate_queue(dir)

            def worker_thread():
                nonlocal jobs_completed
                nonlocal shared_gradient_V, shared_gradient_W 
        
                while True:
                    if work_queue:
                        training_data = work_queue.pop()
                    else:
                        break
                        
                    try:
                        # Get ip, port of available nodes
                        ip, port = self.work_scheduling()
                        print(f"Training model with data: {training_data} at {ip}....")

                        # Attempt compute node connection
                        client, transport = self.connect_compute_node_server(ip, port)

                        # Package new model weights 
                        weights = WeightMatrices(V = model.V.tolist(), W = model.W.tolist())

                        # Train MLP model 
                        gradient = client.trainMLP(weights, training_data, eta, epochs)

                        with mutex:
                            # Update the shared gradient
                            shared_gradient_V = sum_matricies(shared_gradient_V, gradient.V)
                            shared_gradient_W = sum_matricies(shared_gradient_W, gradient.W)
                            jobs_completed += 1

                        transport.close()

                    except Exception as e:
                        print(f"Error in worker: {e}")
                        # Failed job goes back into queue
                        work_queue.append(training_data) 
                        if 'transport' in locals():
                            transport.close()

            threads = []
            num_nodes = len(self.compute_nodes)

            # Get the minimum amount of threads (compute nodes) for jobs
            for i in range(min(num_nodes, len(work_list))):
                thread = threading.Thread(target=worker_thread)
                thread.start()
                threads.append(thread)
            
            # Wait for threads to work through the queue
            for thread in threads:
                thread.join()
            
            if jobs_completed > 0:
                # Average of weights
                shared_gradient_V = scale_matricies(shared_gradient_V, 1.0/jobs_completed)
                shared_gradient_W = scale_matricies(shared_gradient_W, 1.0/jobs_completed)

                # Update model with trained weights
                model.update_weights(shared_gradient_V, shared_gradient_W)
          
            validation_error = model.validate(os.path.join(dir, "validate_letters.txt"))
            print(f"Round {r + 1} validation error: {validation_error}")

        return validation_error
    
def main():
    if len(sys.argv) != 3:
        print("Usage: python coordinator_node.py <port> <scheduling_policy>")
        sys.exit(1)

    port = int(sys.argv[1])
    scheduling_policy = int(sys.argv[2])
    
    if scheduling_policy not in [1, 2]:
        print("Scheduling policy must be 1 (random scheduling) or 2 (load-balancing)")
        sys.exit(1)
    
    handler = CoordinatorHandler(scheduling_policy)
    processor = coordinator.Processor(handler)
    transport = TSocket.TServerSocket(port=port)
    tfactory = TTransport.TBufferedTransportFactory()
    pfactory = TBinaryProtocol.TBinaryProtocolFactory()

    server = TServer.TSimpleServer(processor, transport, tfactory, pfactory)

    print(f"Starting coordinator on port {port} with scheduling policy {scheduling_policy}")
    server.serve()

if __name__ == '__main__':
    main()
