To execute each of the components run the following:

To start compute nodes: 
python3 compute_node.py <port> <load_probability>

For the coordinator node: 
python3 coordinator_node.py <port> <scheduling_policy>

For the client: 
python3 client.py <coordinator_ip> <coordinator_port>

Additionally, ensure the Thrift installation is within PA1's parent directory so:
- /parent
    - /parent/PA1
    - /parent/thrift-0.19.0