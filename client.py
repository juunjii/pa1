#!/usr/bin/env python

#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements. See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership. The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License. You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied. See the License for the
# specific language governing permissions and limitations
# under the License.
#

#
# Largely copied from thrift-0.19.0/tutorial/py/pyClient.py
# 

##
##  CHANGEME  
##  Change the sys.path.insert to where your thrift python libraries are installed
##  Should be in your thrift directory: thrift-0.19.0/lib/py/build/lib*
##
import sys
import glob
import time
sys.path.append('gen-py')
sys.path.insert(0, glob.glob('../thrift-0.19.0/lib/py/build/lib*')[0])
# sys.path.insert(0, "/home/cheh0011/.local/lib/python3.x/site-packages")

## 
##  Make sure to import the service and types 
##  defined in the thrift file
## 


from thrift import Thrift
from thrift.transport import TSocket
from thrift.transport import TTransport
from thrift.protocol import TBinaryProtocol

from compute_node import *
from compute import compute
from compute.ttypes import WeightMatrices
from coordinator import coordinator
from ML import *


def main():
    if len(sys.argv) != 6:
        print("Usage: python3 client.py <coordinator_ip> <coordinator_port> <dir_path> <rounds> <epochs>")
        sys.exit(1)

    # Parse command line arguments
    coordinator_ip = sys.argv[1]
    coordinator_port = int(sys.argv[2])
    dir_path = sys.argv[3]
    rounds = int(sys.argv[4])
    epochs = int(sys.argv[5])

    # Parameters for training MLP
    H = 20  # number of hidden units
    K = 26  # number of possible outcomes (26 letters)
    eta = 0.0001  # learning rate

    try: 
        # Make socket
        transport = TSocket.TSocket(coordinator_ip, coordinator_port)
        transport = TTransport.TBufferedTransport(transport)
        protocol = TBinaryProtocol.TBinaryProtocol(transport)

        client = coordinator.Client(protocol)

        # Connect to coordinator node 
        transport.open()

        print("\nTraining in progress...")

        # Call the train function on the coordinator
        validation_error = client.train(dir_path, rounds, epochs, H, K, eta)

        print(f"\nTraining complete!")
        print(f"Final model validation error: {validation_error:.4f}")

        # Close!
        transport.close()

    except Exception as e:
        print(f"Error: {e}")
        if 'transport' in locals():
            transport.close()
        sys.exit(1)


if __name__ == '__main__':
    try:
        main()
    except Thrift.TException as tx:
        print('%s' % tx.message)