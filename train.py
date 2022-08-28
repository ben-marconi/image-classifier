import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb
import torch
import torch.nn.functional as F
from collections import OrderedDict
from PIL import Image
from torch import nn, optim
from torchvision import datasets, transforms, models


def get_input_args():
    """
    Retrieves and parses command line arguments provided by the user when they run the program from a terminal window. This function uses Python's argparse module to created and define the command line arguments. If the user fails to provide some or all of the arguments, then the default values are used for the missing arguments. 
    Command Line Arguments:
      1. Data directory
      2. Save directory to set directory to save checkpoints as --save_dir, default = 'checkpoint.pth'
      3. CNN Model Architecture as --arch with default value 'vgg'
    Command Line Hyperparameters
      4. Learning rate with default value 0.001
      5. Hidden units in imported network with default value 16
      6. Epochs with default value of 2
      7. GPU mode to select whether GPU or CPU is used
    This function returns these arguments as an ArgumentParser object.
    Parameters:
     None - simply using argparse module to create & store command line arguments
    Returns:
     parse_args() -data structure that stores the command line arguments object  
    """
    
    parser = argparse.ArgumentParser()
    parser.add_argument("datadir", help = 'path to the folder containing the images to train and test images, default is flowers')
    parser.add_argument('--save_dir', type = str, default = 'checkpoint.pth', help = 'file where trained model is saved, default is checkpoint.pth')
    parser.add_argument('--arch', type = str, default = 'vgg', help = 'type of convolutional neural network, default is vgg')
    parser.add_argument('--learning_rate', type = int, default =0.001, help = 'learning rate for training the model, default is 0.001')
    parser.add_argument('--hidden_units', type = int, default = 16, help = 'number of hidden units of the imported model, default is 16')
    parser.add_argument('--epochs', type = int, default = 2, help = 'number of epochs the model is trained on, default is 2')
    parser.add_argument('--gpu', action='store_true', help = 'use GPU to train network instead of CPU')
    return parser.parse_args()
                        
in_arg = get_input_args()
                        
                        
data_dir = in_arg.datadir
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

save_dir = in_arg.save_dir
learn_rate = in_arg.learning_rate
network = (in_arg.arch + str(in_arg.hidden_units)).lower()
epochs = in_arg.epochs
use_gpu = in_arg.gpu


# Define the transforms for the training, validation, and testing sets
train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])


validation_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

test_transforms = validation_transforms



# Load the datasets with ImageFolder
train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
validation_data = datasets.ImageFolder(valid_dir, transform=validation_transforms)
test_data = datasets.ImageFolder(test_dir, transform=test_transforms)


# Using the image datasets and the trainforms, define the dataloaders
trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
validationloader = torch.utils.data.DataLoader(validation_data, batch_size=64)
testloader = torch.utils.data.DataLoader(test_data, batch_size=64)


# Load a pretrained network based on what the user defined in the command line
if network[:8] == 'resnet18':
    model = models.resnet18(pretrained=True)
    type = 'fc'
    output = 512
elif network[:8] == 'resnet34':
    model = models.resnet34(pretrained=True)
    type = 'fc'
    output = 512
elif network[:8] == 'resnet50':
    model = models.resnet50(pretrained=True)
    type = 'fc'
    output = 2048
elif network[:9] == 'resnet101':
    model = models.resnet101(pretrained=True)
    type = 'fc'
    output = 2048
elif network[:9] == 'resnet152':
    model = models.resnet152(pretrained=True)
    type = 'fc'
    output = 512
elif network[:5] == 'vgg11':
    model = models.vgg11(pretrained=True)
    type = 'vgg'
    output = 25088
elif network[:5] == 'vgg13':
    model = models.vgg13(pretrained=True)
    type = 'vgg'
    output = 25088
elif network[:5] == 'vgg16':
    model = models.vgg16(pretrained=True)
    type = 'vgg'
    output = 25088
elif network[:5] == 'vgg19':
    type = 'vgg'
    output = 25088
    model = models.vgg19(pretrained=True)
elif network[:11] == 'densenet121':
    model = models.densenet121(pretrained=True)
    output = 1024
    type = 'classifier'
elif network[:11] == 'densenet161':
    model = models.densenet161(pretrained=True)
    output = 2208
    type = 'classifier'
elif network[:11] == 'densenet169':
    model = models.densenet169(pretrained=True)
    output = 1664
    type = 'classifier'
elif network[:11] == 'densenet201':
    model = models.densenet201(pretrained=True)
    output = 1920
    type = 'classifier'
elif network[:7] == 'alexnet':
    model = models.alexnet(pretrained=True)
    type = 'classifier'
    output = 9216
elif network[:10] == 'squeezenet':
    model = models.squeezenet1_1(pretrained=True)
    type = 'squeezenet'
    output = 13
else:
    model = models.vgg16(pretrained=True)
    type = 'vgg'
    output = 25088
    print("Seleceted network architecture unavailable, vgg16 was chosen instead as default")


# Freeze parameters so we don't backprop through them
for param in model.parameters():
    param.requires_grad = False
    

# Define a new, untrained feed-forward network as a classifier, using ReLU activations and dropout
if type == 'fc':
    model.fc = nn.Sequential(nn.Linear(output, 512),
                                nn.ReLU(),
                                nn.Dropout(0.2),
                                nn.Linear(512, 256),
                                nn.ReLU(),
                                nn.Dropout(0.2),
                                nn.Linear(256, 102),
                                nn.LogSoftmax(dim=1))
    optimizer = optim.Adam(model.fc.parameters(), lr = learn_rate)
    
elif type == 'vgg':
    model.classifier = nn.Sequential(nn.Linear(25088, 512),
                                nn.ReLU(),
                                nn.Dropout(0.2),
                                nn.Linear(512, 102),
                                nn.LogSoftmax(dim=1))
    optimizer = optim.Adam(model.classifier.parameters(), lr = learn_rate)
elif type == 'classifier':
    model.classifier = nn.Sequential(nn.Linear(output, 512),
                                nn.ReLU(),
                                nn.Dropout(0.2),
                                nn.Linear(512, 256),
                                nn.ReLU(),
                                nn.Dropout(0.2),
                                nn.Linear(256, 102),
                                nn.LogSoftmax(dim=1))
    optimizer = optim.Adam(model.classifier.parameters(), lr = learn_rate)
elif type == 'squeezenet':
    model.classifier = nn.Sequential(nn.Linear(output, 1024),
                                nn.ReLU(),
                                nn.Dropout(0.2),
                                nn.Linear(1024, 512),
                                nn.ReLU(),
                                nn.Dropout(0.2),
                                nn.Linear(512, 102),
                                nn.LogSoftmax(dim=1))
    optimizer = optim.Adam(model.classifier.parameters(), lr = learn_rate)    


# Preparations for training
if use_gpu:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")

#Set loss function to be used
criterion = nn.NLLLoss()

model.to(device);


#Train the network
steps = 0
running_loss = 0
print_every = 5

for epoch in range(epochs):
    running_loss = 0
    for images, labels in trainloader:
        steps+=1
        images, labels = images.to(device), labels.to(device)
        log_ps = model(images)
        loss = criterion(log_ps, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if steps % print_every == 0:
            test_loss = 0
            accuracy = 0
            model.eval()
            with torch.no_grad():
                for inputs, labels in validationloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    logps = model.forward(inputs)
                    batch_loss = criterion(logps, labels)
                    
                    test_loss += batch_loss.item()
                    
                    # Calculate accuracy
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    
            print(f"Epoch {epoch+1}/{epochs}.. "
                  f"Train loss: {running_loss/print_every:.3f}.. "
                  f"Validation loss: {test_loss/len(validationloader):.3f}.. "
                  f"Test accuracy: {accuracy/len(validationloader):.3f}")
            running_loss = 0
            model.train()


# Do validation on the test set
accuracy = 0
test_loss = 0
model.eval()

with torch.no_grad():
    for inputs, labels in testloader:
        inputs, labels = inputs.to(device), labels.to(device)
        logps = model.forward(inputs)
        batch_loss = criterion(logps, labels)

        test_loss += batch_loss.item()

        # Calculate accuracy
        ps = torch.exp(logps)
        top_p, top_class = ps.topk(1, dim=1)
        equals = top_class == labels.view(*top_class.shape)
        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
print(f"Test accuracy: {100*accuracy/len(testloader):.1f}%")




# Now that the network is trained, save the model so it can be loaded later for making predictions

checkpoint = {'input' : 25088, 
              'output' : 102,
              'hidden_layers' : [25088,512],
              'optimizer': optimizer.state_dict(),
              'class_to_idx' : train_data.class_to_idx,
              'state_dict' : model.state_dict(),
              'network' : network,
              'output' : output}

torch.save(checkpoint, save_dir)



