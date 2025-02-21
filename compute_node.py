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

class ComputeNodeHandler:
    def __init__(self, load_probability):
        self.load_probability = float(load_probability)

    def trainMLP(self, weights, data, eta, epochs):
        pass

    def loadInjection(self):
        if random.random() < self.load_probability:
            time.sleep(3)  # 3-second delay
    
    def rejectTask(self):
        return random.random() < self.load_probability 


