import socket
import threading
import msvcrt
import os
import torch
import torch.nn as nn
import warnings
import time
import numpy as np

from wisepaasdatahubedgesdk.EdgeAgent import EdgeAgent
import wisepaasdatahubedgesdk.Common.Constants as constant
from wisepaasdatahubedgesdk.Model.Edge import EdgeAgentOptions, MQTTOptions, DCCSOptions, EdgeData, EdgeTag, EdgeStatus, EdgeDeviceStatus, EdgeConfig, NodeConfig, DeviceConfig, AnalogTagConfig, DiscreteTagConfig, TextTagConfig

warnings.filterwarnings("ignore")


def on_connected(edgeAgent, isConnected):
    print("connected !")
    config = __generateConfig()
    _edgeAgent.uploadConfig(action=constant.ActionType['Create'], edgeConfig=config)

def on_disconnected(edgeAgent, isDisconnected):
    print("disconnected !")

def edgeAgent_on_message(agent, messageReceivedEventArgs):
    print("edgeAgent_on_message !")


def __generateConfig():
    config = EdgeConfig()
    deviceConfig = DeviceConfig(id='Server',
                                name='Server',
                                description='Server',
                                deviceType='Server',
                                retentionPolicyName='')

    discrete = DiscreteTagConfig(name='SendGradientToFinalServer',
                                 description='SendGradientToFinalServer',
                                 readOnly=False,
                                 arraySize=0,
                                 state0='Stop',
                                 state1='Start')
    deviceConfig.discreteTagList.append(discrete)

    config.node.deviceList.append(deviceConfig)
    return config

def edge_init():
    _edgeAgent = None
    edgeAgentOptions = EdgeAgentOptions(nodeId='9f101d35-32a1-4d9a-8166-97d5ef340371')
    edgeAgentOptions.connectType = constant.ConnectType['DCCS']
    dccsOptions = DCCSOptions(apiUrl='https://api-dccs-ensaas.wise-paas.iotcenter.nycu.edu.tw/',
                              credentialKey='010ba570af027043436dafadf61352x4')
    edgeAgentOptions.DCCS = dccsOptions
    _edgeAgent = EdgeAgent(edgeAgentOptions)
    _edgeAgent.on_connected = on_connected
    _edgeAgent.on_disconnected = on_disconnected
    _edgeAgent.on_message = edgeAgent_on_message
    _edgeAgent.connect()
    time.sleep(5)  # Waiting for connection to be established
    return _edgeAgent

# Define the neural network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(18, 12)
        self.fc2 = nn.Linear(12, 6)
        self.fc3 = nn.Linear(6, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x

# This function checks for 'q' key press
def check_for_q_key():
    global stop_server
    while True:
        if msvcrt.kbhit():
            ch = msvcrt.getch().decode('utf-8')
            if ch == 'q':  # 'q' key
                stop_server = True
                break

# This function handles a connection with a client
def handle_client(c, addr):
    # Determine the file name based on the client's IP address
    if addr[0] == '192.168.0.125':
        filename = 'gradients1.pth'
    elif addr[0] == '192.168.0.168':
        filename = 'gradients2.pth'
    elif addr[0] == '192.168.0.171':
        filename = 'gradients3.pth'
    else:
        filename = 'gradients.pth'  # Default file name

    # Receive the file size from the client
    size = int(c.recv(1024).decode('utf-8'))

    # Send an acknowledgment to the client
    c.send('Ready to receive file'.encode())
    print('Message sent to client!')

    # Receive the file
    with open(filename, 'wb') as f:
        while size > 0:
            data = c.recv(min(1024, size))
            if not data:
                break
            f.write(data)
            size -= len(data)
    print('File received from client!')

    # Close the connection with the client
    c.close()

# This function sends the model.pth file to the client
def send_file_to_client():
    # Connect to the client
    client_socket1 = socket.socket()
    client_socket2 = socket.socket()
    client_socket3 = socket.socket()
    client_socket1.connect(('192.168.0.125', 12346))
    client_socket2.connect(('192.168.0.168', 12347))
    client_socket3.connect(('192.168.0.171', 12348))


    # Open the file in read-binary mode
    # with open('avg_grad.pth', 'rb') as f:
    with open('model.pth', 'rb') as f:
        data = f.read()

    # Send the file size to the client
    client_socket1.send(str(len(data)).encode())
    client_socket2.send(str(len(data)).encode())
    client_socket3.send(str(len(data)).encode())

    # Wait for an acknowledgment from the client
    ack1 = client_socket1.recv(1024).decode()
    ack2 = client_socket2.recv(1024).decode()
    ack3 = client_socket3.recv(1024).decode()
    if ack1 == 'Ready to receive file':
        # Send the file
        client_socket1.sendall(data)
    else:
        print('Failed to send file to client!')

    if ack2 == 'Ready to receive file':
        # Send the file
        client_socket2.sendall(data)
    else:
        print('Failed to send file to client!')

    if ack3 == 'Ready to receive file':
        # Send the file
        client_socket3.sendall(data)
    else:
        print('Failed to send file to client!')

    # Close the connection
    client_socket1.close()
    client_socket2.close()
    client_socket3.close()

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Initialize the edge device
_edgeAgent = edge_init()

# Create a socket object
s = socket.socket()

# Set SO_REUSEADDR option
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

# Define the port on which you want to connect
port = 12345

# Bind to the port
s.bind(('', port))

# Put the socket into listening mode
s.listen(1)
print('Server is listening')

# This flag will be used to stop the server
stop_server = False

# Start a new thread that checks for 'q' key press
threading.Thread(target=check_for_q_key).start()

# Initialize the neural network
set_seed(12345)
net = Net()
torch.save(net.state_dict(), 'model.pth')

# Initialize the last modification times to 0
last_modified1 = 0
last_modified2 = 0
last_modified3 = 0

# A forever loop until we interrupt it or an error occurs
while True:
    if stop_server:
        print('Stopping server')
        break
    # Establish connection with client
    try:
        s.settimeout(10)  # Set a timeout for the accept function
        c, addr = s.accept()
        print('Got Connection from:', addr)

        # Start a new thread that handles the connection
        threading.Thread(target=handle_client, args=(c, addr)).start()
        time.sleep(1)
    except socket.timeout:
        continue

    # Get the current modification times
    try:
        current_modified1 = os.path.getmtime('gradients1.pth')
    except FileNotFoundError:
        print("File 'gradients1.pth' not found. Waiting for the file to be created...")
        continue

    try:
        current_modified2 = os.path.getmtime('gradients2.pth')
    except FileNotFoundError:
        print("File 'gradients2.pth' not found. Waiting for the file to be created...")
        continue

    try:
        current_modified3 = os.path.getmtime('gradients3.pth')
    except FileNotFoundError:
        print("File 'gradients3.pth' not found. Waiting for the file to be created...")
        continue

    # Check if all three files are updated
    if current_modified1 > last_modified1 and current_modified2 > last_modified2 and current_modified3 > last_modified3:
        # Load the gradients from each file
        gradients1 = torch.load('gradients1.pth')
        gradients2 = torch.load('gradients2.pth')
        gradients3 = torch.load('gradients3.pth')

        # Aggregate gradients from all Raspberry Pis
        for name, param in net.named_parameters():
            avg_grad = (gradients1[name] + gradients2[name] + gradients3[name]) / 3
            param.grad = avg_grad

        # Save gradients
        gradients = {name: param.grad for name, param in net.named_parameters()}
        torch.save(gradients, 'avg_grad.pth')

        optimizer = torch.optim.Adam(net.parameters(), lr=0.01)

        # Update parameters with averaged gradients
        optimizer.step()

        # Save the updated model
        torch.save(net.state_dict(), 'model.pth')

        optimizer.zero_grad()

        # Update the last modification times
        last_modified1 = current_modified1
        last_modified2 = current_modified2
        last_modified3 = current_modified3

        # Send avg_grad.pth to client
        send_file_to_client()

        # Construct an EdgeData object
        data = EdgeData()
        # Add a single tag to this data object, setting the value to 1
        data.tagList.append(EdgeTag(deviceId='Server', tagName='SendGradientToFinalServer', value=1))
        # Use the EdgeAgent instance to send this data to the cloud platform
        _edgeAgent.sendData(data)

    # Sleep for a while to reduce CPU usage
    time.sleep(1)

s.close()
