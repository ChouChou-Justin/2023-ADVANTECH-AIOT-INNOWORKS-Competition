# Import necessary libraries
import torch
import torch.nn as nn
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
import socket
import os
import time
import csv
import numpy as np

from wisepaasdatahubedgesdk.EdgeAgent import EdgeAgent
import wisepaasdatahubedgesdk.Common.Constants as constant
from wisepaasdatahubedgesdk.Model.Edge import EdgeAgentOptions, MQTTOptions, DCCSOptions, EdgeData, EdgeTag, EdgeStatus, \
    EdgeDeviceStatus, EdgeConfig, NodeConfig, DeviceConfig, AnalogTagConfig, DiscreteTagConfig, TextTagConfig


def on_connected(edgeAgent, isConnected):
    print("connected !")
    config = __generateConfig()
    edgeAgent.uploadConfig(action=constant.ActionType['Create'], edgeConfig=config)


def on_disconnected(edgeAgent, isDisconnected):
    print("disconnected !")


def edgeAgent_on_message(agent, messageReceivedEventArgs):
    print("edgeAgent_on_message !")


def __generateConfig():
    config = EdgeConfig()
    deviceConfig = DeviceConfig(id='RaspberryPi3',
                                name='RaspberryPi3',
                                description='RaspberryPi3',
                                deviceType='RaspberryPi',
                                retentionPolicyName='')

    discrete1 = DiscreteTagConfig(name='SendGradientToMidServer',
                                  description='SendGradientToMidServer',
                                  readOnly=False,
                                  arraySize=0,
                                  state0='Stop',
                                  state1='Start')
    deviceConfig.discreteTagList.append(discrete1)

    discrete2 = DiscreteTagConfig(name='Alarm',
                                  description='Alarm',
                                  readOnly=False,
                                  arraySize=0,
                                  state0='Stop',
                                  state1='Start')
    deviceConfig.discreteTagList.append(discrete2)

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
    time.sleep(3)  # Waiting for connection to be established
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


# Function to start a server to receive model.pth
def start_server_for_file():
    # Create a socket object
    ss = socket.socket()
    ss.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    # Define the port on which you want to connect
    receive_port = 12348

    # Bind to the port
    ss.bind(('', receive_port))

    # Put the socket into listening mode
    ss.listen(1)
    print('Client is listening for model.pth')

    while True:
        # Establish connection with server
        sc, addr = ss.accept()
        print('Got Connection from:', addr)

        # Receive the file size from the server
        size = int(sc.recv(1024).decode('utf-8'))

        # Send an acknowledgment to the server
        sc.send('Ready to receive file'.encode())
        print('Message sent to server!')

        # Receive the file
        with open('model.pth', 'wb') as f:
            while size > 0:
                data = sc.recv(min(1024, size))
                if not data:
                    break
                f.write(data)
                size -= len(data)
        print('File received from server!')

        # Close the connection with the server
        sc.close()
        break  # Once file is received, break the loop and end the function

    # Close the server
    ss.close()


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    # Initialize the edge device
    _edgeAgent = edge_init()

    # Load heart_prep.csv dataset
    data = pd.read_csv('heart_prep.csv')

    # Use all features for each Raspberry Pi
    X = data.drop('HeartDisease', axis=1).values
    y = data['HeartDisease'].values

    # Split the training data
    X_train = X[540:810]
    y_train = y[540:810]

    X_test = X[810:]
    y_test = y[810:]

    # Convert to PyTorch tensors
    X_train = torch.FloatTensor(X_train)
    X_test = torch.FloatTensor(X_test)
    y_train = torch.FloatTensor(y_train)
    y_test = torch.FloatTensor(y_test)

    # Initialize the neural network
    set_seed(12345)
    net = Net()
    torch.save(net.state_dict(), 'model.pth')

    # Define loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)

    acc_list = []
    std = 0
    epochs = 5

    # Train the neural network on one sample at a time
    for epoch in tqdm(range(epochs)):
        print('Epoch:', epoch + 1)
        net.train()  # Set the model back to training mode
        for i in range(len(X_train)):
            # Forward pass
            y_pred = net(X_train[i])
            loss = criterion(y_pred, y_train[i].unsqueeze(0))

            if (y_pred > 0.5):
                # Construct an EdgeData object
                data = EdgeData()
                # Add a single tag to this data object, setting the value to 1
                data.tagList.append(EdgeTag(deviceId='RaspberryPi3', tagName='Alarm', value=1))
                # Use the EdgeAgent instance to send this data to the cloud platform
                _edgeAgent.sendData(data)
            else:
                # Construct an EdgeData object
                data = EdgeData()
                # Add a single tag to this data object, setting the value to 0
                data.tagList.append(EdgeTag(deviceId='RaspberryPi3', tagName='Alarm', value=0))
                # Use the EdgeAgent instance to send this data to the cloud platform
                _edgeAgent.sendData(data)

            # Backward pass
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)

            net.fc1.weight.grad = net.fc1.weight.grad + (torch.randn_like(net.fc1.weight.grad) * std)
            net.fc2.weight.grad = net.fc2.weight.grad + (torch.randn_like(net.fc2.weight.grad) * std)
            net.fc3.weight.grad = net.fc3.weight.grad + (torch.randn_like(net.fc3.weight.grad) * std)

            # Save gradients
            gradients = {name: param.grad for name, param in net.named_parameters()}
            torch.save(gradients, 'gradients3.pth')

            # Create a socket object
            s = socket.socket()

            # Define the port on which you want to connect
            port = 12345

            # Construct an EdgeData object
            data = EdgeData()
            # Add a single tag to this data object, setting the value to 1
            data.tagList.append(EdgeTag(deviceId='RaspberryPi3', tagName='SendGradientToMidServer', value=0))
            # Use the EdgeAgent instance to send this data to the cloud platform
            _edgeAgent.sendData(data)

            try:
                # Connect to the server on local computer
                s.connect(('192.168.0.133', port))
                print('Server Connected!')

                # Get the size of the file
                size = os.path.getsize('gradients3.pth')

                # Send the file size to the server
                s.send(str(size).encode())

                # Wait for the server to acknowledge
                print('Message received from server:', s.recv(1024).decode('utf-8'))

                # Send the file
                with open('gradients3.pth', 'rb') as f:
                    for _ in range(size):
                        bytes_read = f.read(1024)
                        if not bytes_read:
                            break
                        s.sendall(bytes_read)
                print('File sent to server!')

                # Construct an EdgeData object
                data = EdgeData()
                # Add a single tag to this data object, setting the value to 1
                data.tagList.append(EdgeTag(deviceId='RaspberryPi3', tagName='SendGradientToMidServer', value=1))
                # Use the EdgeAgent instance to send this data to the cloud platform
                _edgeAgent.sendData(data)



            except Exception as e:
                print('File reception failed:', str(e))

            # Close the connection
            s.close()

            # Call start_server_for_file function after sending gradients file to the server
            start_server_for_file()

            # Load gradients and model
            net.load_state_dict(torch.load('model.pth'))

            # Zero out the current gradients
            optimizer.zero_grad()

        # Test the model after training 1 epoch
        net.load_state_dict(torch.load('model.pth'))  # Load the trained model
        net.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            outputs = net(X_test)
            predicted = (outputs > 0.5).float().squeeze()
            total = y_test.size(0)
            correct = (predicted == y_test).sum().item()

        accuracy = correct / total * 100
        acc_list.append(accuracy)

    print('Acc of the model on the testing data for each epoch', acc_list)

    # Save the accuracy to a csv file
    with open('accuracy_for_std_{}.csv'.format(std), 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['STD', 'Epoch', 'Accuracy'])
        for i, acc in enumerate(acc_list, start=1):
            writer.writerow([std, i, acc])


if __name__ == "__main__":
    main()

