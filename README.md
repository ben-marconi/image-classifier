# image-classifier

This project is a command-line app in which a user can train a neural network by inputting a directory with image files for training, testing and validating. 
train.py and predict.py both have mandatory and optional arguments which are inserted from the command line. These are explained below:

## train.py
### Command Line Arguments:
  1. Data directory
  2. Save directory to set directory to save checkpoints as --save_dir, default = 'checkpoint.pth'
  3. CNN Model Architecture as --arch with default value 'vgg'
  Command Line Hyperparameters
  4. Learning rate with default value 0.001
  5. Hidden units in imported network with default value 16
  6. Epochs with default value of 2
  7. GPU mode to select whether GPU or CPU is used



## predict.py
### Command Line Arguments:
  1. Image file path to be predicted
  2. Directory from which saved model is loaded, default = 'checkpoint.pth'
  3. Topk, returns the top 'k' most probable classes as --topk, default = 5
  4. Mapping of categories to real names as --category_names default = cat_to_name.json
  5. GPU mode to select whether GPU or CPU is used





##



For more information on the arguments run the following command:
```
python train.py --help
```

The default image directory is a folder called 'flowers'. 
The file checkpoint.pth contains a pretrained neuralnetwork which can be used when predicting the content of a new image. 
