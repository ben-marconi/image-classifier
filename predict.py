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
      1. Image file path to be predicted
      2. Directory from which saved model is loaded, default = 'checkpoint.pth'
      3. Topk, returns the top 'k' most probable classes as --topk, default = 5
      4. Mapping of categories to real names as --category_names default = cat_to_name.json
      5. GPU mode to select whether GPU or CPU is used
    This function returns these arguments as an ArgumentParser object.
    Parameters:
     None - simply using argparse module to create & store command line arguments
    Returns:
     parse_args() -data structure that stores the command line arguments object  
    """
    
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help = 'path to the file containing the image to be predicted')
    parser.add_argument("checkpoint", help = 'file containing the model that is used to make the prediction')
    parser.add_argument("--category_names", type = str, default = 'cat_to_name.json', help = 'file that maps categories to real names')
    parser.add_argument("--topk", type = int, default = 5, help = 'returns the top \'k\' most probable classes, default is 5')
    parser.add_argument('--gpu', action='store_true', help = 'use GPU to train network instead of CPU')
    return parser.parse_args()


in_arg = get_input_args()



image_path = in_arg.image
checkpoint = in_arg.checkpoint
cat_to_name_file = in_arg.category_names
topk = in_arg.topk
use_gpu = in_arg.gpu


# Set device based on user preference
if use_gpu:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")


with open(cat_to_name_file, 'r') as f:
    cat_to_name = json.load(f)
  

    
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    network = checkpoint['network']
    output = checkpoint['output']
    if network[:6] == 'resnet':
        classifier = nn.Sequential(nn.Linear(output, 512),
                                    nn.ReLU(),
                                    nn.Dropout(0.2),
                                    nn.Linear(512, 256),
                                    nn.ReLU(),
                                    nn.Dropout(0.2),
                                    nn.Linear(256, 102),
                                    nn.LogSoftmax(dim=1))
        
    elif network[:3] == 'vgg':
        classifier = nn.Sequential(nn.Linear(output, 512),
                                    nn.ReLU(),
                                    nn.Dropout(0.2),
                                    nn.Linear(512, 102),
                                    nn.LogSoftmax(dim=1))

    elif network == 'classifier':
        classifier = nn.Sequential(nn.Linear(output, 512),
                                    nn.ReLU(),
                                    nn.Dropout(0.2),
                                    nn.Linear(512, 256),
                                    nn.ReLU(),
                                    nn.Dropout(0.2),
                                    nn.Linear(256, 102),
                                    nn.LogSoftmax(dim=1))
    elif network[:10] == 'squeezenet':
        classifier = nn.Sequential(nn.Linear(output, 1024),
                                    nn.ReLU(),
                                    nn.Dropout(0.2),
                                    nn.Linear(1024, 512),
                                    nn.ReLU(),
                                    nn.Dropout(0.2),
                                    nn.Linear(512, 102),
                                    nn.LogSoftmax(dim=1))
    if network == 'resnet18':
        model = models.resnet18(pretrained=True)
    elif network == 'resnet34':
        model = models.resnet34(pretrained=True)
    elif network == 'resnet50':
        model = models.resnet50(pretrained=True)
    elif network == 'resnet101':
        model = models.resnet101(pretrained=True)
    elif network == 'resnet152':
        model = models.resnet152(pretrained=True)
    elif network == 'vgg11':
        model = models.vgg11(pretrained=True)
    elif network == 'vgg13':
        model = models.vgg13(pretrained=True)
    elif network == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif network == 'vgg19':
        model = models.vgg19(pretrained=True)
    elif network == 'densenet121':
        model = models.densenet121(pretrained=True)
    elif network == 'densenet161':
        model = models.densenet161(pretrained=True)
    elif network == 'densenet169':
        model = models.densenet169(pretrained=True)
    elif network == 'densenet201':
        model = models.densenet201(pretrained=True)
    elif network[:7] == 'alexnet':
        model = models.alexnet(pretrained=True)
    elif network[:10] == 'squeezenet':
        model = models.squeezenet1_1(pretrained=True)
    else:
        model = models.vgg16(pretrained=True)
    model.classifier = classifier
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    model.to(device);
    return model

model = load_checkpoint(checkpoint)



def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    with Image.open(image) as im:
        im = im.resize((256, 256))
        im = im.crop((16, 16, 240, 240))
        np_image = np.array(im)
        means = [0.485, 0.456, 0.406]
        stdevs = [0.229, 0.224, 0.225]
        np_image = np_image/255
        np_normalised = (np_image-means)/stdevs
        img = np_normalised.transpose((2, 0, 1))
        return img


    
def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))
    ax.axis('off')
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image);
    
    return ax



def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    image = torch.from_numpy(process_image(image_path))
    image = torch.unsqueeze(image, 0)
    image = image.float()
    image = image.to(device)
    model.to(device)
    model.eval()
    logps = model(image)
    ps = torch.exp(logps)
    top_p, top_classes = ps.topk(topk, dim=1)
    flower_probs = OrderedDict()
    class_to_idx = model.class_to_idx
    idx_to_class = {v:k for k,v in class_to_idx.items()}

    for i in range(topk):
        flower_class = idx_to_class[top_classes[0][i].item()]
        class_name = cat_to_name[flower_class]
        flower_probs[class_name] = top_p[0][i].item()
    return flower_probs


# Function that displays the probabilities of the topk classes in a seaborn plot
probs = predict(image_path, model, topk)

print("===========================================================================")
print("                 Predictions for submitted Image: \n")
print("===========================================================================")
i = 1
for name, probability in probs.items():
    print(f"{i}. {name.capitalize()}               Probability: {probability*100:.1f}%\n")
    i += 1
    
print("===========================================================================")
