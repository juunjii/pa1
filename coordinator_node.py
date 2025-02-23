import sys
import os
import threading
import random
from queue import Queue
from thrift import Thrift
from thrift.transport import TSocket
from thrift.transport import TTransport
from thrift.protocol import TBinaryProtocol
from thrift.server import TServer

# from coordinator import coordinator
# from coordinator.ttypes import WeightMatrices
from compute import compute
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
    def connet_compute_node_server(self, host, port):
        try:
            transport = TSocket.TSocket(host, port)
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
   
    def train(self, dir, rounds, epochs, h, k, eta):
        

if __name__ == '__main__':
    obj = CoordinatorHandler(1)

    obj.parse_compute_nodes()