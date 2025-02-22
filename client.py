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
from MLP import *

def main():
    # obj = ComputeNodeHandler()

    # weights = WeightMatrices(
    #         V=[[0.1, 0.2], [0.3, 0.4]],  # Example V matrix
    #         W=[[0.5, 0.6], [0.7, 0.8]]   # Example W matrix
    # )

    # file = "letters/train_letters1.txt"
    # obj.trainMLP(weights, file, 0.001, 15)

    # # read ip and port from command line
    # if len(sys.argv) < 3:
    #     print("python3 ex_client.py <ip> <port>")
    #     return
    # ip = sys.argv[1]
    # port = sys.argv[2]

    # # Thrift client boilerplate
    # # Enjoy the enthusiastic comments included by the thrift developers

    # # Make socket
    # transport = TSocket.TSocket(ip, port)

    # # Buffering is critical. Raw sockets are very slow
    # transport = TTransport.TBufferedTransport(transport)

    # # Wrap in a protocol
    # protocol = TBinaryProtocol.TBinaryProtocol(transport)

    # # Create a client to use the protocol encoder
    # client = compute.Client(protocol)

    # # Connect!
    # transport.open()

    # client.ping()
    
    # # Simple input loop to read a string from the client
    # # Calls "add_the" on the server and gets a response
    # usr_input = ""
    # while usr_input != "exit":
        
    #     print("Enter a string to modify")
        
    #     usr_input = input(">> ")
    #     if usr_input == "exit":
    #         break

    #     # return type is of strint()
    #     _response = client.add_the(usr_input)
    #     print("Server response [%d letters]: %s"%(_response.number, _response.words))

    # # Close!
    # transport.close()


if __name__ == '__main__':
    try:
        main()
    except Thrift.TException as tx:
        print('%s' % tx.message)