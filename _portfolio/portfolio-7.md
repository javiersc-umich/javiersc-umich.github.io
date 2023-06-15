---
title: "Autoencoders & CLIP models"
excerpt: "Implemented Autoencoder model for image compression and representational learning. Moreover, the CLIP model is explored for query tasks as shown below. Press blue link above for more details.<br/><img src='/images/autoencoders.png'> <br/><img src='/images/clip.png'>"
collection: portfolio
---
# Starting

Run the following code to import the modules you'll need. After your finish the assignment, remember to run all cells and save the note book to your local machine as a .ipynb file for Canvas submission.

### Google Colab Setup
We need to run a few commands to set up our environment on Google Colab. If you are running this notebook on a local machine you can skip this section.

Run the following cell to mount your Google Drive. Follow the link, sign in to your Google account (the same account you used to store this notebook!) and copy the authorization code into the text box that appears below.


```
from google.colab import drive
drive.mount('/content/drive')
```

    Mounted at /content/drive



```

!pip install torchsummary
!pip install transformers
import pickle
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import cv2
import time
import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

import torchvision
from torchvision import datasets, models, transforms
from torchsummary import summary

import albumentations as A
import matplotlib.pyplot as plt
from transformers import DistilBertModel, DistilBertConfig, DistilBertTokenizer
from tqdm import tqdm_notebook
from tqdm.autonotebook import tqdm

print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)
# Detect if we have a GPU available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print("Using the GPU!")
else:
    print("WARNING: Could not find GPU! Using CPU only. If you want to enable GPU, please to go Edit > Notebook Settings > Hardware Accelerator and select GPU.")

np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
```

    Looking in indexes: https://pypi.org/simple, https://us-python.pkg.dev/colab-wheels/public/simple/
    Requirement already satisfied: torchsummary in /usr/local/lib/python3.7/dist-packages (1.5.1)
    Looking in indexes: https://pypi.org/simple, https://us-python.pkg.dev/colab-wheels/public/simple/
    Collecting transformers
      Downloading transformers-4.24.0-py3-none-any.whl (5.5 MB)
    [K     |████████████████████████████████| 5.5 MB 14.4 MB/s
    [?25hCollecting huggingface-hub<1.0,>=0.10.0
      Downloading huggingface_hub-0.11.0-py3-none-any.whl (182 kB)
    [K     |████████████████████████████████| 182 kB 62.5 MB/s
    [?25hCollecting tokenizers!=0.11.3,<0.14,>=0.11.1
      Downloading tokenizers-0.13.2-cp37-cp37m-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (7.6 MB)
    [K     |████████████████████████████████| 7.6 MB 46.1 MB/s
    [?25hRequirement already satisfied: numpy>=1.17 in /usr/local/lib/python3.7/dist-packages (from transformers) (1.21.6)
    Requirement already satisfied: importlib-metadata in /usr/local/lib/python3.7/dist-packages (from transformers) (4.13.0)
    Requirement already satisfied: regex!=2019.12.17 in /usr/local/lib/python3.7/dist-packages (from transformers) (2022.6.2)
    Requirement already satisfied: packaging>=20.0 in /usr/local/lib/python3.7/dist-packages (from transformers) (21.3)
    Requirement already satisfied: filelock in /usr/local/lib/python3.7/dist-packages (from transformers) (3.8.0)
    Requirement already satisfied: pyyaml>=5.1 in /usr/local/lib/python3.7/dist-packages (from transformers) (6.0)
    Requirement already satisfied: tqdm>=4.27 in /usr/local/lib/python3.7/dist-packages (from transformers) (4.64.1)
    Requirement already satisfied: requests in /usr/local/lib/python3.7/dist-packages (from transformers) (2.23.0)
    Requirement already satisfied: typing-extensions>=3.7.4.3 in /usr/local/lib/python3.7/dist-packages (from huggingface-hub<1.0,>=0.10.0->transformers) (4.1.1)
    Requirement already satisfied: pyparsing!=3.0.5,>=2.0.2 in /usr/local/lib/python3.7/dist-packages (from packaging>=20.0->transformers) (3.0.9)
    Requirement already satisfied: zipp>=0.5 in /usr/local/lib/python3.7/dist-packages (from importlib-metadata->transformers) (3.10.0)
    Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /usr/local/lib/python3.7/dist-packages (from requests->transformers) (1.24.3)
    Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.7/dist-packages (from requests->transformers) (2022.9.24)
    Requirement already satisfied: idna<3,>=2.5 in /usr/local/lib/python3.7/dist-packages (from requests->transformers) (2.10)
    Requirement already satisfied: chardet<4,>=3.0.2 in /usr/local/lib/python3.7/dist-packages (from requests->transformers) (3.0.4)
    Installing collected packages: tokenizers, huggingface-hub, transformers
    Successfully installed huggingface-hub-0.11.0 tokenizers-0.13.2 transformers-4.24.0
    PyTorch Version:  1.12.1+cu113
    Torchvision Version:  0.13.1+cu113
    Using the GPU!


# PS 9. Self-supervised learning

In this problem, we are going to implement two representation learning methods: an autoencoder and a recent constrastive learning method.
We'll then test the features that were learned by these models on a "downstream" recognition task, using the STL-10 dataset.


# Downloading the dataset.

We use PyTorch built-in class to download  the STL-10 (http://ai.stanford.edu/~acoates/stl10/) dataset (a subset of ImageNet). The STL-10 dataset contains three partitions: train, test, and unlabeled. The train partition contains 10 image classes, each class with 500 images. The test partition contains 800 images for each class. The unlabeled contains a total of 100,000 images with many classes not in the train/test partitions.


```
unlabeled_transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                         transforms.RandomCrop(64),
                                         transforms.ToTensor()])

labeled_transform = transforms.Compose([transforms.CenterCrop(64),
                                        transforms.ToTensor(),])


# We use the PyTorch built-in class to download the STL-10 dataset.
# The 'unlabeled' partition contains 100,000 images without labels.
# It's used for leanring representations with unsupervised learning.
dataset_un = torchvision.datasets.STL10('./data', 'unlabeled', download=True, transform=unlabeled_transform)

dataset_tr = torchvision.datasets.STL10('./data', 'train', download=False, transform=labeled_transform)
dataset_te = torchvision.datasets.STL10('./data', 'test', download=False, transform=labeled_transform)
```

    Downloading http://ai.stanford.edu/~acoates/stl10/stl10_binary.tar.gz to ./data/stl10_binary.tar.gz



      0%|          | 0/2640397119 [00:00<?, ?it/s]


    Extracting ./data/stl10_binary.tar.gz to ./data



```
print('# of samples for ulabeled, train, and test, {}, {}, {}'.format(len(dataset_un), len(dataset_tr), len(dataset_te)))
print('Classes in train: {}'.format(dataset_tr.classes))
```

    # of samples for ulabeled, train, and test, 100000, 5000, 8000
    Classes in train: ['airplane', 'bird', 'car', 'cat', 'deer', 'dog', 'horse', 'monkey', 'ship', 'truck']



```
# Visualize the data within the dataset
class_names = dict(zip(range(10), dataset_tr.classes))
dataloader_un = DataLoader(dataset_un, batch_size=64)
dataloader_tr = DataLoader(dataset_tr, batch_size=64)

def imshow(inp, title=None, ax=None, figsize=(5, 5)):
  """Imshow for Tensor."""
  inp = inp.numpy().transpose((1, 2, 0))
  if ax is None:
    fig, ax = plt.subplots(1, figsize=figsize)
  ax.imshow(inp)
  ax.set_xticks([])
  ax.set_yticks([])
  if title is not None:
    ax.set_title(title)

# Visualize training partition
# Get a batch of training data
inputs, classes = next(iter(dataloader_tr))

# Make a grid from batch
out = torchvision.utils.make_grid(inputs, nrow=8)

fig, ax = plt.subplots(1, figsize=(10, 10))
title = [class_names[x.item()] if (i+1) % 8 != 0 else class_names[x.item()]+'\n' for i, x in enumerate(classes)]
imshow(out, title=' | '.join(title), ax=ax)

# Visualize unlabeled partition
inputs, classes = next(iter(dataloader_un))
out = torchvision.utils.make_grid(inputs, nrow=8)

fig, ax = plt.subplots(1, figsize=(10, 10))
imshow(out, title='unlabeled', ax=ax)
```



![png](/images/autoencoders_files/autoencoders_9_0.png)





![png](/images/autoencoders_files/autoencoders_9_1.png)



As can be seen from above visualizations, the unlabeled partition contains classes that are not in the training partition. Though not labeled, the unlabeled partition has much more data than the labeled training partition. The large amount of unlabeled label ought to help us learn useful representations. In the next sections, we will use the unlabeled partition to help learn representations that is helpful for downstream tasks.

# Part 1. Autoencoders

We will first build an autoencoder. To keep training time low, we'll use a very simple network structure.

## 1.1 Build the encoder
Please make sure that your encoder has the same architeture as we print below before your proceed to the decoder part.

### Encoder archiecture

----------------------------------------------------------------
        Layer (type)               Output Shape         Param #  
            Conv2d-1           [-1, 12, 32, 32]             588  
              ReLU-2           [-1, 12, 32, 32]               0  
            Conv2d-3           [-1, 24, 16, 16]           4,632  
              ReLU-4           [-1, 24, 16, 16]               0  
            Conv2d-5             [-1, 48, 8, 8]          18,480  
              ReLU-6             [-1, 48, 8, 8]               0   
            Conv2d-7             [-1, 24, 4, 4]          18,456  
              ReLU-8             [-1, 24, 4, 4]               0  
Total params: 42,156  
Trainable params: 42,156  
Non-trainable params: 0  
Input size (MB): 0.05
Forward/backward pass size (MB): 0.33  
Params size (MB): 0.16  
Estimated Total Size (MB): 0.54  


```
class Encoder(nn.Module):
    def __init__(self, in_channels=3):
        super(Encoder, self).__init__()

        ##############################################################################
        #                               YOUR CODE HERE                               #
        ##############################################################################
        # TODO: Build an encoder with the architecture as specified above.           #
        ##############################################################################
        conv1 = nn.Conv2d(in_channels, 12, kernel_size=4,stride=2,padding=1)
        activate = nn.ReLU()
        conv2 = nn.Conv2d(12, 24, kernel_size=4, stride=2,padding=1)
        conv3 = nn.Conv2d(24,48, kernel_size=4, stride=2, padding=1)
        conv4 = nn.Conv2d(48,24, kernel_size=4, stride=2, padding=1)
        self.encoder = nn.Sequential(conv1, activate, conv2, activate, conv3, activate, conv4, activate)
        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################

    def forward(self, x):
        '''
        Given an image x, return the encoded latent representation h.

        Args:
            x: torch.tensor

        Return:
            h: torch.tensor
        '''

        h = self.encoder(x)

        return h
```


```
# Print out the neural network architectures and activation dimensions.
# Verify that your network has the same architecture as the one we printed above.
encoder = Encoder().to(device)
summary(encoder, [(3, 64, 64)])
```

    ----------------------------------------------------------------
            Layer (type)               Output Shape         Param #
    ================================================================
                Conv2d-1           [-1, 12, 32, 32]             588
                  ReLU-2           [-1, 12, 32, 32]               0
                Conv2d-3           [-1, 24, 16, 16]           4,632
                  ReLU-4           [-1, 24, 16, 16]               0
                Conv2d-5             [-1, 48, 8, 8]          18,480
                  ReLU-6             [-1, 48, 8, 8]               0
                Conv2d-7             [-1, 24, 4, 4]          18,456
                  ReLU-8             [-1, 24, 4, 4]               0
    ================================================================
    Total params: 42,156
    Trainable params: 42,156
    Non-trainable params: 0
    ----------------------------------------------------------------
    Input size (MB): 0.05
    Forward/backward pass size (MB): 0.33
    Params size (MB): 0.16
    Estimated Total Size (MB): 0.54
    ----------------------------------------------------------------


## 1.2 Build the decoder

Next, we build the decoder to reconstruct the image from the latent representation extracted by the encoder. Please implement the decoder following the architectrue printed here.

----------------------------------------------------------------
        Layer (type)               Output Shape         Param #  
              ConvTranspose2d-1   [-1, 48, 8, 8]         18,480  
              ReLU-2              [-1, 48, 8, 8]              0  
              ConvTranspose2d-3   [-1, 24, 16, 16]       18,456  
              ReLU-4              [-1, 24, 16, 16]            0
              ConvTranspose2d-5   [-1, 12, 32, 32]        4,620  
              ReLU-6              [-1, 12, 32, 32]            0  
              ConvTranspose2d-7   [-1, 3, 64, 64]           579  
              Sigmoid-8           [-1, 3, 64, 64]             0  

Total params: 42,135  
Trainable params: 42,135  
Non-trainable params: 0  
Input size (MB): 0.00  
Forward/backward pass size (MB): 0.52  
Params size (MB): 0.16  
Estimated Total Size (MB): 0.68  


```
class Decoder(nn.Module):
    def __init__(self, out_channels=3, feat_dim=64):
        super(Decoder, self).__init__()

        ##############################################################################
        #                               YOUR CODE HERE                               #
        ##############################################################################
        # TODO: Build the decoder as specified above.                                #
        ##############################################################################
        deconv1 = nn.ConvTranspose2d(24, 48, kernel_size=4, stride=2, padding=1)
        activate = nn.ReLU()
        deconv2 = nn.ConvTranspose2d(48, 24, kernel_size=4, stride=2, padding=1)
        deconv3 = nn.ConvTranspose2d(24, 12, kernel_size=4, stride=2, padding=1)
        deconv4 = nn.ConvTranspose2d(12, out_channels, kernel_size=4, stride=2, padding=1)
        sig = nn.Sigmoid()
        self.decoder = nn.Sequential(deconv1, activate, deconv2, activate, deconv3, activate, deconv4, sig)
        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################

    def forward(self, h):
        '''
        Given latent representation h, reconstruct an image patch of size 64 x 64.

        Args:
            h: torch.tensor

        Return:
            x: torch.tensor
        '''
        x = self.decoder(h)
        return x
```


```
# Print out the neural network architectures and activation dimensions.
# Verify that your network has the same architecture as the one we printed above.
decoder = Decoder().to(device)
summary(decoder, [(24, 4, 4)])
```

    ----------------------------------------------------------------
            Layer (type)               Output Shape         Param #
    ================================================================
       ConvTranspose2d-1             [-1, 48, 8, 8]          18,480
                  ReLU-2             [-1, 48, 8, 8]               0
       ConvTranspose2d-3           [-1, 24, 16, 16]          18,456
                  ReLU-4           [-1, 24, 16, 16]               0
       ConvTranspose2d-5           [-1, 12, 32, 32]           4,620
                  ReLU-6           [-1, 12, 32, 32]               0
       ConvTranspose2d-7            [-1, 3, 64, 64]             579
               Sigmoid-8            [-1, 3, 64, 64]               0
    ================================================================
    Total params: 42,135
    Trainable params: 42,135
    Non-trainable params: 0
    ----------------------------------------------------------------
    Input size (MB): 0.00
    Forward/backward pass size (MB): 0.52
    Params size (MB): 0.16
    Estimated Total Size (MB): 0.68
    ----------------------------------------------------------------


## 1.3 Put together the autoencoder

Now we have the encoder and the decoder classes. We only need to implement another `Autoencoder` class to wrap the encoder and the decoder together.


```
class Autoencoder(nn.Module):
    def __init__(self, in_channels=3, feat_dim=64):
        super(Autoencoder, self).__init__()

        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        '''
        Compress and reconstruct the input image with encoder and decoder.

        Args:
            x: torch.tensor

        Return:
            x_: torch.tensor
        '''

        h = self.encoder(x)
        x_ = self.decoder(h)

        return x_
```


```
# verify that your aueconder's output size is 3 x 64 x 64
ae = Autoencoder().to(device)
summary(ae, (3, 64, 64))
```

    ----------------------------------------------------------------
            Layer (type)               Output Shape         Param #
    ================================================================
                Conv2d-1           [-1, 12, 32, 32]             588
                  ReLU-2           [-1, 12, 32, 32]               0
                Conv2d-3           [-1, 24, 16, 16]           4,632
                  ReLU-4           [-1, 24, 16, 16]               0
                Conv2d-5             [-1, 48, 8, 8]          18,480
                  ReLU-6             [-1, 48, 8, 8]               0
                Conv2d-7             [-1, 24, 4, 4]          18,456
                  ReLU-8             [-1, 24, 4, 4]               0
               Encoder-9             [-1, 24, 4, 4]               0
      ConvTranspose2d-10             [-1, 48, 8, 8]          18,480
                 ReLU-11             [-1, 48, 8, 8]               0
      ConvTranspose2d-12           [-1, 24, 16, 16]          18,456
                 ReLU-13           [-1, 24, 16, 16]               0
      ConvTranspose2d-14           [-1, 12, 32, 32]           4,620
                 ReLU-15           [-1, 12, 32, 32]               0
      ConvTranspose2d-16            [-1, 3, 64, 64]             579
              Sigmoid-17            [-1, 3, 64, 64]               0
              Decoder-18            [-1, 3, 64, 64]               0
    ================================================================
    Total params: 84,291
    Trainable params: 84,291
    Non-trainable params: 0
    ----------------------------------------------------------------
    Input size (MB): 0.05
    Forward/backward pass size (MB): 0.95
    Params size (MB): 0.32
    Estimated Total Size (MB): 1.31
    ----------------------------------------------------------------


## 1.4 Training the autoencoder

Now, we'll train the autoencoder to reconstruct images from the unlabeled set of STL-10. Note that the reconstructed images will contain significant artifacts, due to the limited size of the bottleneck between the encoder and decoder, and the small network size.


```
# We train on 10,000 unsupervised samples instead of 100,000 samples to speed up training
n = 10000
dataset_un_subset, _ = torch.utils.data.random_split(dataset_un, [n,100000-n])
dataloader_un = DataLoader(dataset_un_subset, batch_size=128, shuffle=True)
dataloader_tr = DataLoader(dataset_tr, batch_size=128, shuffle=True)
dataloader_te = DataLoader(dataset_te, batch_size=128, shuffle=False)
```


```
def visualize_recon(model, dataloader):
    '''
    Helper function for visualizing reconstruction performance.

    Randomly sample 8 images and plot the original/reconstructed images.
    '''
    model.eval()
    img = next(iter(dataloader))[0][:8].to(device)
    out = model(img)

    fig, ax = plt.subplots(1, 1, figsize=(15,10))
    inp = torchvision.utils.make_grid(torch.cat((img, out), dim=2), nrow=8)
    imshow(inp.detach().cpu(), ax=ax)
    model.train()
    plt.show()
```


```
def train_ae(model, dataloader, epochs=200):
    '''
    Train autoencoder model.

    Args:
        model: torch.nn.module.
        dataloader: DataLoader. The unlabeled partition of the STL dataset.
    '''

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    loss_traj = []

    for epoch in tqdm_notebook(range(epochs)):

        loss_epoch = 0
        for x, _ in dataloader:

            ##############################################################################
            #                               YOUR CODE HERE                               #
            ##############################################################################
            # TODO: Train the autoencoder on one minibatch.                              #
            ##############################################################################
            optimizer.zero_grad()
            x = x.to(device)
            output = model(x)
            loss = criterion(output, x)
            loss.backward()
            optimizer.step()
            ##############################################################################
            #                               END OF YOUR CODE                             #
            ##############################################################################
            loss_epoch += loss.detach()

        loss_traj.append(loss_epoch)

        if epoch % 10 == 0:
            print('Epoch {}, loss {:.3f}'.format(epoch, loss_epoch))
            visualize_recon(model, dataloader)

    return model, loss_traj
```


```
# Train the autoencoder for 100 epochs
ae = Autoencoder().to(device)
ae, ae_loss_traj = train_ae(ae, dataloader_un, epochs=100)
torch.save(ae.state_dict(), 'ae.pth')
```

    /usr/local/lib/python3.7/dist-packages/ipykernel_launcher.py:14: TqdmDeprecationWarning: This function will be removed in tqdm==5.0.0
    Please use `tqdm.notebook.tqdm` instead of `tqdm.tqdm_notebook`




      0%|          | 0/100 [00:00<?, ?it/s]


    Epoch 0, loss 4.511




![png](/images/autoencoders_files/autoencoders_26_3.png)



    Epoch 10, loss 1.260




![png](/images/autoencoders_files/autoencoders_26_5.png)



    Epoch 20, loss 0.923




![png](/images/autoencoders_files/autoencoders_26_7.png)



    Epoch 30, loss 0.832




![png](/images/autoencoders_files/autoencoders_26_9.png)



    Epoch 40, loss 0.755




![png](/images/autoencoders_files/autoencoders_26_11.png)



    Epoch 50, loss 0.714




![png](/images/autoencoders_files/autoencoders_26_13.png)



    Epoch 60, loss 0.686




![png](/images/autoencoders_files/autoencoders_26_15.png)



    Epoch 70, loss 0.666




![png](/images/autoencoders_files/autoencoders_26_17.png)



    Epoch 80, loss 0.655




![png](/images/autoencoders_files/autoencoders_26_19.png)



    Epoch 90, loss 0.640




![png](/images/autoencoders_files/autoencoders_26_21.png)



After training the autoencoder on the 100,000 images for 100 epochs, we see that autoencoder has leanred to approximately recontruct the image.

## 1.5 Train a linear classifier

Now, we ask how useful the features are for object recongition. We'll train a linear clasifier that takes the output of the encoder as its features. During training, we freeze the parameters of the encoder. To verify the effectiveness of unsupervised pretraining, we compare the linear classifier accuracy against two baselines:

* Supervised: train the encoder together with the linear classifier on the training set for 100 epochs.
* Random weights: freeze the parameters of a randomly initialized encoder during training.


```
# latent representation dimension (the output dimension of the encoder)
feat_dim = 24 * 4 * 4
```


```
def train_classifier(encoder, cls, dataloader, epochs=100, supervised=False):
    '''
    Args:
        encoder: trained/untrained encoder for unsupervised/supervised training.
        cls: linear classifier.
        dataloader: train partition.
        supervised:

    Return:
        cls: linear clssifier.
    '''

    optimizer = optim.Adam(cls.parameters(), lr=0.001, weight_decay=1e-4)
    if supervised:
        optimizer = optim.Adam(list(cls.parameters())+list(encoder.parameters()), lr=0.001, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    loss_traj = []
    accuracy_traj = []

    for epoch in tqdm_notebook(range(epochs)):

        loss_epoch = 0
        corrects_epoch = 0
        for x, y in dataloader:

            batch_size = x.size(0)
            x = x.float().to(device)
            y = y.to(device)
            ##############################################################################
            #                               YOUR CODE HERE                               #
            ##############################################################################
            # TODO: update the parameters of the classifer. If in supervised mode, the   #
            # parameter of the encoder is also updated.                                  #
            ##############################################################################

            #print(x.shape)

            #print(encoder(x).view(batch_size,-1).shape)

            #batch_features = x.view(batch_size,3*feat_dim)

            #print(batch_features.shape)

            if supervised==True:
              optimizer.zero_grad()
              encoder.train()
              outs = cls(encoder(x).view(batch_size,-1))
              loss = criterion(outs,y)
              loss.backward()
              optimizer.step()
            else:
              optimizer.zero_grad()
              encoder.eval()
              outs = cls(encoder(x).view(batch_size,-1))
              loss = criterion(outs,y)
              loss.backward()
              optimizer.step()

            ##############################################################################
            #                               END OF YOUR CODE                             #
            ##############################################################################
            _, preds = torch.max(outs, 1)
            corrects_epoch += torch.sum(preds == y.data)
            loss_epoch += loss.detach()

        loss_traj.append(loss_epoch)
        epoch_acc = corrects_epoch.double() / len(dataloader.dataset)
        accuracy_traj.append(epoch_acc)

        if epoch % 10 == 0:
            print('Epoch {}, loss {:.3f}, train accuracy {}'.format(epoch, loss_epoch, epoch_acc))

    return cls, loss_traj
```


```
def test(encoder, cls, dataloader):
    '''
    Calculate the accuracy of the trained linear classifier on the test set.
    '''
    cls.eval()

    loss_epoch = 0
    corrects_epoch = 0
    for x, y in dataloader:

        x = x.float()
        batch_size = x.size(0)
        x, y = x.to(device), y.to(device)
        h = encoder(x).view(batch_size, -1)
        outs = cls(h)
        _, preds = torch.max(outs, 1)
        corrects_epoch += torch.sum(preds == y.data)

    epoch_acc = corrects_epoch.double() / len(dataloader.dataset)
    print('Test accuracy {}'.format(epoch_acc))
```


```
# Method I: unsupervised pretraining + training linear classifier
# Freeze the parameters of the trained autoencoder
# Train a linear classifier using the features
linear_cls = nn.Sequential(nn.Linear(feat_dim, 10)).to(device)
cls_unsupervised, loss_traj_unsupervised = train_classifier(ae.encoder, linear_cls, dataloader_tr, epochs=100, supervised=False)
test(ae.encoder, cls_unsupervised, dataloader_te)
```

    /usr/local/lib/python3.7/dist-packages/ipykernel_launcher.py:20: TqdmDeprecationWarning: This function will be removed in tqdm==5.0.0
    Please use `tqdm.notebook.tqdm` instead of `tqdm.tqdm_notebook`



      0%|          | 0/100 [00:00<?, ?it/s]


    Epoch 0, loss 95.280, train accuracy 0.1648
    Epoch 10, loss 76.068, train accuracy 0.3352
    Epoch 20, loss 73.320, train accuracy 0.3638
    Epoch 30, loss 71.900, train accuracy 0.38520000000000004
    Epoch 40, loss 69.966, train accuracy 0.38980000000000004
    Epoch 50, loss 68.898, train accuracy 0.4052
    Epoch 60, loss 68.693, train accuracy 0.4048
    Epoch 70, loss 68.435, train accuracy 0.4046
    Epoch 80, loss 68.194, train accuracy 0.4148
    Epoch 90, loss 69.132, train accuracy 0.39680000000000004
    Test accuracy 0.31975000000000003



```
# Method II: supervised training
# Train the encoder together with the linear classifier
linear_cls = nn.Sequential(nn.Linear(feat_dim, 10)).to(device)
encoder = Autoencoder().to(device).encoder
cls_supervised, loss_traj_supervised = train_classifier(encoder, linear_cls, dataloader_tr, epochs=100, supervised=True)
test(encoder, cls_supervised, dataloader_te)
```

    /usr/local/lib/python3.7/dist-packages/ipykernel_launcher.py:20: TqdmDeprecationWarning: This function will be removed in tqdm==5.0.0
    Please use `tqdm.notebook.tqdm` instead of `tqdm.tqdm_notebook`



      0%|          | 0/100 [00:00<?, ?it/s]


    Epoch 0, loss 89.045, train accuracy 0.1572
    Epoch 10, loss 60.800, train accuracy 0.4288
    Epoch 20, loss 54.022, train accuracy 0.503
    Epoch 30, loss 45.953, train accuracy 0.5800000000000001
    Epoch 40, loss 38.551, train accuracy 0.654
    Epoch 50, loss 29.371, train accuracy 0.7428
    Epoch 60, loss 25.953, train accuracy 0.7764000000000001
    Epoch 70, loss 14.904, train accuracy 0.87
    Epoch 80, loss 10.036, train accuracy 0.9254
    Epoch 90, loss 6.020, train accuracy 0.9602
    Test accuracy 0.435125



```
# Method III: random encoder + training linear classifier
# We freeze the parameters of the randomly initialized encoder during training
linear_cls = nn.Sequential(nn.Linear(feat_dim, 10)).to(device)
encoder = Autoencoder().to(device).encoder
cls_random, loss_traj_random = train_classifier(encoder, linear_cls, dataloader_tr, epochs=100, supervised=False)
test(encoder, cls_random, dataloader_te)
```

    /usr/local/lib/python3.7/dist-packages/ipykernel_launcher.py:20: TqdmDeprecationWarning: This function will be removed in tqdm==5.0.0
    Please use `tqdm.notebook.tqdm` instead of `tqdm.tqdm_notebook`



      0%|          | 0/100 [00:00<?, ?it/s]


    Epoch 0, loss 92.059, train accuracy 0.0956
    Epoch 10, loss 90.893, train accuracy 0.21400000000000002
    Epoch 20, loss 90.082, train accuracy 0.23820000000000002
    Epoch 30, loss 89.410, train accuracy 0.24600000000000002
    Epoch 40, loss 88.795, train accuracy 0.2534
    Epoch 50, loss 88.439, train accuracy 0.2526
    Epoch 60, loss 88.096, train accuracy 0.2534
    Epoch 70, loss 87.768, train accuracy 0.2528
    Epoch 80, loss 87.487, train accuracy 0.2534
    Epoch 90, loss 87.289, train accuracy 0.2526
    Test accuracy 0.25825


With pretrained encoder, the linear classifier should achieve about 30% accuracy on the test set. With the supervised approach, the linear classifier should achieve an accuracy above 40%. The random encoder approach performs the worse among these three. The observation that the pretrained encoder outperforms the random encoder confirms that unsupervised pretraining has learned a useful represenation. However, the quality of this learned representation is not good because it only performs slightly better than the random encoder. In the next part, we'll explore contrastive multiview coding, which learns a more useful representation than the autoencoder.


```
del dataset_un
del dataset_te
del dataset_tr
```

# Part 2. Contrastive Language-Image Pre-training (CLIP)




## Setup
As the authors suggest, in CLIP the dataset plays a major role for the performance and we need both image and texts for training the CLIP model. We will use [flickr8k](https://www.kaggle.com/datasets/adityajn105/flickr8k) to train our model. This is a small dataset with 8000 images, which would run well given the limited comuptation units on Colab. Each image in the dataset is paired with 5 captions, we will just use one of the 5 captions for simplicity. Below we displayed some sample captions corresponding to an image.



```
!wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1csewExbHtYIcOr8BWMrGd8DqX_awmP1t' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1csewExbHtYIcOr8BWMrGd8DqX_awmP1t" -O flickr8k.zip && rm -rf /tmp/cookies.txt
!unzip flickr8k.zip
```

    [1;30;43mStreaming output truncated to the last 5000 lines.[0m
      inflating: captions.txt            



```

# dataset pre-processing
df = pd.read_csv("captions.txt")
img_color = cv2.imread('/content/Images/'+ str(df['image'][0]),1)
plt.imshow(cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB))
df['id'] = [id_ for id_ in range(df.shape[0] // 5) for _ in range(5)]
df.to_csv("captions.csv", index=False)
df = pd.read_csv("captions.csv")
df.head()

```





  <div id="df-1331677f-9463-45c4-98f2-5adda344ce6c">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>image</th>
      <th>caption</th>
      <th>id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1000268201_693b08cb0e.jpg</td>
      <td>A child in a pink dress is climbing up a set o...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1000268201_693b08cb0e.jpg</td>
      <td>A girl going into a wooden building .</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1000268201_693b08cb0e.jpg</td>
      <td>A little girl climbing into a wooden playhouse .</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1000268201_693b08cb0e.jpg</td>
      <td>A little girl climbing the stairs to her playh...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1000268201_693b08cb0e.jpg</td>
      <td>A little girl in a pink dress going into a woo...</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-1331677f-9463-45c4-98f2-5adda344ce6c')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-1331677f-9463-45c4-98f2-5adda344ce6c button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-1331677f-9463-45c4-98f2-5adda344ce6c');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>






![png](/images/autoencoders_files/autoencoders_40_1.png)



Here are some configuration variables, hyperparameter and utility functions that we will be using through this section:


```
image_path = "/content/Images"
captions_path = "/content"
batch_size = 32
num_workers = 2
head_lr = 1e-3
image_encoder_lr = 1e-4
text_encoder_lr = 1e-5
weight_decay = 1e-3
patience = 1
factor = 0.8
epochs = 3

image_encoder_model = 'resnet50'
image_embedding = 2048
text_encoder_model = "distilbert-base-uncased"
text_embedding = 768
text_tokenizer = "distilbert-base-uncased"
max_length = 200

pretrained = True # for both image encoder and text encoder
trainable = True # for both image encoder and text encoder
temperature = 0.07

image_size = 64

# for projection head; used for both image and text encoders
projection_dim = 256

class AvgMeter:
    def __init__(self, name="Metric"):
        self.name = name
        self.reset()

    def reset(self):
        self.avg, self.sum, self.count = [0] * 3

    def update(self, val, count=1):
        self.count += count
        self.sum += val * count
        self.avg = self.sum / self.count

    def __repr__(self):
        text = f"{self.name}: {self.avg:.4f}"
        return text

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]
```

## Dataset

Below we are writing the custom dataloader that can handle both the images and texts necessary to train the CLIP model. It can be noted from the CLIP figure that, we need to encode both images and their describing texts. Since, we are not going to feed raw text to our text encoder! We will use the [DistilBERT](https://huggingface.co/docs/transformers/model_doc/distilbert) (which is smaller than BERT but performs nearly as well as BERT) model as our text encoder. Before that, we need to tokenize the sentences (captions) with DistilBERT tokenizer and then feed the token ids (input_ids) and the attention masks to DistilBERT. We have implemented the tokenization scheme and you do not have to implement anything for this custom dataloader.



```
class CLIPDataset(torch.utils.data.Dataset):
    def __init__(self, image_filenames, captions, tokenizer, transforms):
        """
        image_filenames and cpations must have the same length; so, if there are
        multiple captions for each image, the image_filenames must have repetitive
        file names
        """

        self.image_filenames = image_filenames
        self.captions = list(captions)
        self.encoded_captions = tokenizer(
            list(captions), padding=True, truncation=True, max_length=max_length
        )
        self.transforms = transforms

    def __getitem__(self, idx):
        item = {
            key: torch.tensor(values[idx])
            for key, values in self.encoded_captions.items()
        }

        image = cv2.imread(f"{image_path}/{self.image_filenames[idx]}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transforms(image=image)['image']
        item['image'] = torch.tensor(image).permute(2, 0, 1).float()
        item['caption'] = self.captions[idx]

        return item


    def __len__(self):
        return len(self.captions)

# data augmentation for images
def get_transforms(mode="train"):
    return A.Compose(
            [
                A.Resize(image_size, image_size, always_apply=True),
                A.Normalize(max_pixel_value=255.0, always_apply=True),
            ]
        )
```

## Image Encoder
We use [ResNet50](https://pytorch.org/vision/main/models/generated/torchvision.models.resnet50.html) as our image encoder. It encodes each image to a fixed size vector of the model's output channels (in the case of ResNet50, it's 2048). We will load the pre-trained ImageNet weights to the ResNet50 model to get faster convergence and the limitation on the dataset size and computation resources.


```
from torchvision.models.resnet import ResNet50_Weights
class ImageEncoder(nn.Module):
    """
    Encode images to a fixed size vector
    """

    def __init__(
        self, model_name=image_encoder_model, pretrained=pretrained, trainable=trainable
    ):
        super().__init__()
        self.model = torchvision.models.resnet50(ResNet50_Weights.IMAGENET1K_V2)
        self.model.fc = nn.Identity()

        for p in self.model.parameters():
            p.requires_grad = trainable

    def forward(self, x):
        return self.model(x)
```

## Text Encoder
As mentioned above, we use DistilBERT as our text encoder. The output representation is a vector of size 768.


```
class TextEncoder(nn.Module):
    def __init__(self, model_name=text_encoder_model, pretrained=pretrained, trainable=trainable):
        super().__init__()
        if pretrained:
            self.model = DistilBertModel.from_pretrained(model_name)
        else:
            self.model = DistilBertModel(config=DistilBertConfig())

        for p in self.model.parameters():
            p.requires_grad = trainable

        # we are using the CLS token hidden representation as the sentence's embedding
        self.target_token_idx = 0

    def forward(self, input_ids, attention_mask):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = output.last_hidden_state
        return last_hidden_state[:, self.target_token_idx, :]
```

## Projection Head

Now that we have encoded both our images and texts into fixed size vectors (2048 for image and 768 for text), we need to project both embeddings into a new vector space (!) with similar dimensions (256). This allows for both images and texts to be able to compare them and push apart the non-relevant image and texts and pull together those that match.


```
class ProjectionHead(nn.Module):
    def __init__(
        self,
        embedding_dim,
        projection_dim=projection_dim,
    ):
        super().__init__()
        '''
        Args:
            embedding_dim (int): Extracted Image or text feature embedding dimenasion.
        '''

        ##############################################################################
        #                               YOUR CODE HERE                               #
        ##############################################################################
        # TODO: Initialize a single layer linear transformation for the projection   #
        # head.                                                                      #
        ##############################################################################
        self.project = nn.Linear(embedding_dim, projection_dim)
        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################

    def forward(self, x):
        '''
        Args:
            x: Image or text feature embeddings extracted from the ResNet50 and DistilBERT model respectively.

        Return:
            projected: The projected image and text embeddings.
        '''
        ##############################################################################
        #                               YOUR CODE HERE                               #
        ##############################################################################
        # TODO: Write the forward function and normalize the out of the projection   #
        # head. hint: use F.normalize() for the normalization                        #
        ##############################################################################

        projected = self.project(x)

        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################
        return projected
```

## CLIP
In this section, you will need to use the modules that you just implemented to build the CLIP model. The loss function we minimize is defined as:


$$
\mathcal{L}_{\text {contrast }}^{I, T}=-\frac{1}{N}\sum_{i=1}^N \log \frac{\exp(I_{i}\cdot T_{i}/\tau)}{\sum_{j=1}^{N} \exp(I_{i}\cdot T_{j}/\tau)},
$$


where $I_i$ and $T_i$ are the image feature and its corresponding text representations for the $i_{th}$ sample in a minibatch, and $N$ denotes the batch size. The constant $\tau$ is used to control the range of logits.



We will minimize a symmetric objective function that averages $\mathcal{L}_{\text {contrast }}^{I, T}$ and $\mathcal{L}_{\text {contrast }}^{T, I}$, i.e.,



$$\mathcal{L}=\frac{\mathcal{L}_{\text {contrast }}^{I, T}+\mathcal{L}_{\text {contrast }}^{T, I}}{2},
$$

where,
$$% \mathcal{L}_{\text {contrast }}^{I, T}=-\frac{1}{N}\sum_{i=1}^N \log \frac{\exp(I_{i}\cdot T_{i})}{\sum_{j=1}^{N} \exp(I_{i}\cdot T_{j})}\cdot\exp(\tau),
\mathcal{L}_{\text {contrast }}^{T, I}=-\frac{1}{N}\sum_{i=1}^N \log \frac{\exp(T_{i}\cdot I_{i}/\tau)}{\sum_{j=1}^{N} \exp(T_{i}\cdot I_{j}/\tau)},
$$

By minimizing the above loss function, we learn representations for the image and text such that the positive pairs (respective images and their captions) will produce high dot product while giving low dot product for negative samples (image and other unrelated captions).




```
class CLIPModel(nn.Module):
    def __init__(
        self,
        temperature=temperature,
        image_embedding=image_embedding,
        text_embedding=text_embedding,
    ):
        super().__init__()
        '''
        Args:
            temperature (float): temperature parameter which controls the range of the logits.
            image_embedding (int): Shape of the extracted image embedding
            text_embedding (int): Shape of the extracted text embedding

        '''

        self.image_encoder = None
        self.text_encoder = None
        self.image_projection = None
        self.text_projection = None
        self.temperature = temperature
        ##############################################################################
        #                               YOUR CODE HERE                               #
        ##############################################################################
        # TODO: Initialize the encoders and the projection heads for image and text i.e,
        # instantiate the above None variables with their corresponding models#
        ##############################################################################
        self.image_encoder =  ImageEncoder()
        self.text_encoder =  TextEncoder()
        self.image_projection = ProjectionHead(image_embedding)
        self.text_projection = ProjectionHead(text_embedding)
        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################


    def forward(self, batch):

        '''
        Args:
            batch: batch of images for training.

        Return:
            loss: computed loss.
        '''
        # get image and text features
        image_features = self.image_encoder(batch["image"])
        text_features = self.text_encoder(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        )

        loss = None
        ##############################################################################
        #                               YOUR CODE HERE                               #
        ##############################################################################
        # TODO: Project images and text into a new vector space and write the loss function by following the above equations
        # you are not allowed to use any for loops while computing the loss.
        ##############################################################################
        textFeatures = self.text_projection(text_features)
        imageFeatures = self.image_projection(image_features)
        target = torch.arange(textFeatures.shape[0], device=textFeatures.get_device())
        loss1 = F.cross_entropy(self.temperature*(imageFeatures @ textFeatures.T),target)
        loss2 = F.cross_entropy(self.temperature*(textFeatures @ imageFeatures.T),target)
        loss = 0.5*(loss1 + loss2)
        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################

        return loss.mean()

```

## Training
Next we will need to train the CLIP model and here are some utility functions that are necessary to train.


```
def make_train_valid_dfs():
    dataframe = pd.read_csv(f"{captions_path}/captions.csv")
    dataframe = dataframe[dataframe.reset_index().index % 5 == 0]
    max_id = dataframe["id"].max() + 1
    image_ids = np.arange(0, max_id)
    np.random.seed(42)
    valid_ids = np.random.choice(
        image_ids, size=int(0.2 * len(image_ids)), replace=False
    )
    train_ids = [id_ for id_ in image_ids if id_ not in valid_ids]
    train_dataframe = dataframe[dataframe["id"].isin(train_ids)].reset_index(drop=True)
    valid_dataframe = dataframe[dataframe["id"].isin(valid_ids)].reset_index(drop=True)
    return train_dataframe, valid_dataframe


def build_loaders(dataframe, tokenizer, mode):
    transforms = get_transforms(mode=mode)
    dataset = CLIPDataset(
        dataframe["image"].values,
        dataframe["caption"].values,
        tokenizer=tokenizer,
        transforms=transforms,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True if mode == "train" else False,
        drop_last=True
    )
    return dataloader

def train_epoch(model, train_loader, optimizer, lr_scheduler, step):
    loss_meter = AvgMeter()
    tqdm_object = tqdm(train_loader, total=len(train_loader))
    for i, batch in enumerate(tqdm_object):
        batch = {k: v.to(device) for k, v in batch.items() if k != "caption"}
        loss = model(batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step == "batch":
            lr_scheduler.step()

        count = batch["image"].size(0)
        loss_meter.update(loss.item(), count)

        tqdm_object.set_postfix(train_loss=loss_meter.avg, lr=get_lr(optimizer))

        if i % 100 == 0:
            print('loss:', loss.item() / count)
    return loss_meter


def valid_epoch(model, valid_loader):
    loss_meter = AvgMeter()

    tqdm_object = tqdm(valid_loader, total=len(valid_loader))
    for batch in tqdm_object:
        batch = {k: v.to(device) for k, v in batch.items() if k != "caption"}
        loss = model(batch)

        count = batch["image"].size(0)
        loss_meter.update(loss.item(), count)

        tqdm_object.set_postfix(valid_loss=loss_meter.avg)
    return loss_meter
```

Run this cell to train the model.


```
train_df, valid_df = make_train_valid_dfs()
tokenizer = DistilBertTokenizer.from_pretrained(text_tokenizer)
train_loader = build_loaders(train_df, tokenizer, mode="train")
valid_loader = build_loaders(valid_df, tokenizer, mode="valid")


model = CLIPModel().to(device)
params = [
    {"params": model.image_encoder.parameters(), "lr": image_encoder_lr},
    {"params": model.text_encoder.parameters(), "lr": text_encoder_lr},
    {"params": itertools.chain(
        model.image_projection.parameters(), model.text_projection.parameters()
    ), "lr": head_lr, "weight_decay": weight_decay}
]
optimizer = torch.optim.AdamW(params, weight_decay=0.)
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", patience=patience, factor=factor
)
step = "epoch"

best_loss = float('inf')
for epoch in range(epochs):
    print(f"Epoch: {epoch + 1}")
    model.train()
    train_loss = train_epoch(model, train_loader, optimizer, lr_scheduler, step)
    model.eval()
    with torch.no_grad():
        valid_loss = valid_epoch(model, valid_loader)

    if valid_loss.avg < best_loss:
        best_loss = valid_loss.avg
        torch.save(model.state_dict(),  "best.pt")
        print("Saved Best Model!")

    lr_scheduler.step(valid_loss.avg)
```

    Some weights of the model checkpoint at distilbert-base-uncased were not used when initializing DistilBertModel: ['vocab_projector.weight', 'vocab_transform.bias', 'vocab_projector.bias', 'vocab_transform.weight', 'vocab_layer_norm.bias', 'vocab_layer_norm.weight']
    - This IS expected if you are initializing DistilBertModel from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
    - This IS NOT expected if you are initializing DistilBertModel from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).


    Epoch: 1



      0%|          | 0/202 [00:00<?, ?it/s]


    loss: 0.10859090089797974
    loss: 0.07754897326231003
    loss: 0.06961987912654877



      0%|          | 0/50 [00:00<?, ?it/s]


    Saved Best Model!
    Epoch: 2



      0%|          | 0/202 [00:00<?, ?it/s]


    loss: 0.05206213891506195
    loss: 0.042692847549915314
    loss: 0.04333950951695442



      0%|          | 0/50 [00:00<?, ?it/s]


    Saved Best Model!
    Epoch: 3



      0%|          | 0/202 [00:00<?, ?it/s]


    loss: 0.02990029752254486
    loss: 0.024682089686393738
    loss: 0.04304511845111847



      0%|          | 0/50 [00:00<?, ?it/s]


## Train a Linear Classifier
We'll train a linear clasifier that takes the output of the encoder (Image Encoder: ResNet50) as its features. During training, we freeze the parameters of the encoder and only train the linear layer.


```
labeled_transform = transforms.Compose([
    transforms.CenterCrop(64),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

dataset_tr = torchvision.datasets.STL10('./data', 'train', download=False, transform=labeled_transform)
dataset_te = torchvision.datasets.STL10('./data', 'test', download=False, transform=labeled_transform)
dataloader_tr = DataLoader(dataset_tr, batch_size=128, shuffle=True, num_workers=2)
dataloader_te = DataLoader(dataset_te, batch_size=128, shuffle=False, num_workers=2)
model = CLIPModel().to(device)
model.load_state_dict(torch.load("best.pt", map_location=device))
model.eval()
encoder = model.image_encoder

linear_cls = nn.Sequential(
    nn.Linear(image_embedding, 10)
    ).to(device)

cls_clip, loss_traj_clip = train_classifier(encoder, linear_cls, dataloader_tr, epochs=100, supervised=False)
test(encoder, cls_clip, dataloader_te)
```

    Some weights of the model checkpoint at distilbert-base-uncased were not used when initializing DistilBertModel: ['vocab_projector.weight', 'vocab_transform.bias', 'vocab_projector.bias', 'vocab_transform.weight', 'vocab_layer_norm.bias', 'vocab_layer_norm.weight']
    - This IS expected if you are initializing DistilBertModel from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
    - This IS NOT expected if you are initializing DistilBertModel from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
    /usr/local/lib/python3.7/dist-packages/ipykernel_launcher.py:20: TqdmDeprecationWarning: This function will be removed in tqdm==5.0.0
    Please use `tqdm.notebook.tqdm` instead of `tqdm.tqdm_notebook`



      0%|          | 0/100 [00:00<?, ?it/s]


    Epoch 0, loss 49.835, train accuracy 0.6528
    Epoch 10, loss 7.360, train accuracy 0.9642000000000001
    Epoch 20, loss 3.731, train accuracy 0.9918
    Epoch 30, loss 2.292, train accuracy 0.9984000000000001
    Epoch 40, loss 1.494, train accuracy 0.9994000000000001
    Epoch 50, loss 1.087, train accuracy 1.0
    Epoch 60, loss 0.832, train accuracy 1.0
    Epoch 70, loss 0.661, train accuracy 1.0
    Epoch 80, loss 0.555, train accuracy 1.0
    Epoch 90, loss 0.465, train accuracy 1.0
    Test accuracy 0.801875


With pretrained encoder, the linear classifier should achieve an accuracy > 75% on the test set. Notice that the supervised flag is set to 'False' which means we are updating the weights of the final linear layer.

# Report results

So far, we have trained linear classifiers on top of autoencoder representations and CLIP representations.

With CLIP, we learn a more useful representation than autoencoder

Please report the test accuracy of all four linear classifiers below.

### Autoencoder-based
method 1 accuracy: 0.31975  
method 2 accuracy: 0.435125  
method 3 accuracy: 0.25825  

### CLIP
accuracy: 0.801875


## CLIP Inference
Now we are doing a interesting downstream task to see what CLIP has learned. We will give the model a piece of text, and let the model retrieve the most relevant images from the dataset.

First, we need to project all the images into a common search space.



```
def get_image_embeddings(df, model_path):
    tokenizer = DistilBertTokenizer.from_pretrained(text_tokenizer)
    dataloader = build_loaders(df, tokenizer, mode="valid")

    model = CLIPModel().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    image_embeddings = []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            image_features = model.image_encoder(batch["image"].to(device))
            image_embedding = model.image_projection(image_features)
            image_embeddings.append(image_embedding)
    return model, torch.cat(image_embeddings)

df, _ = make_train_valid_dfs()
model, image_embeddings = get_image_embeddings(df, "best.pt")
```

    Some weights of the model checkpoint at distilbert-base-uncased were not used when initializing DistilBertModel: ['vocab_projector.weight', 'vocab_transform.bias', 'vocab_projector.bias', 'vocab_transform.weight', 'vocab_layer_norm.bias', 'vocab_layer_norm.weight']
    - This IS expected if you are initializing DistilBertModel from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
    - This IS NOT expected if you are initializing DistilBertModel from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).



      0%|          | 0/202 [00:00<?, ?it/s]


In this function `retrieve_images`, we will get the model, image embeddings and a text query. You need to calculate the similarities between the text and all the images and return the best `k` of them.


```
def retrieve_images(model, image_embeddings, query, image_filenames, k=9):
    tokenizer = DistilBertTokenizer.from_pretrained(text_tokenizer)
    encoded_query = tokenizer([query])
    batch = {
        key: torch.tensor(values).to(device)
        for key, values in encoded_query.items()
    }
    with torch.no_grad():
        text_features = model.text_encoder(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        )
        text_embeddings = model.text_projection(text_features)


    ##############################################################################
    #                               YOUR CODE HERE                               #
    ##############################################################################
    # TODO: Please get the top k similar images to the given text query and pass them to the 'matches' variable.
    # Note: You cannot use any for loops in this function.
    ##############################################################################
    matches = None # Please keep this as the variable name for the matches
    results = (image_embeddings @ text_embeddings.T)
    _, indices = torch.sort(results, dim=0, descending=True)
    indices = indices[0:k,0].cpu().numpy()
    matches = image_filenames[indices]

    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

    _, axes = plt.subplots(3, 3, figsize=(10, 10))
    for match, ax in zip(matches, axes.flatten()):
        image = cv2.imread(f"{image_path}/{match}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ax.imshow(image)
        ax.axis("off")

    plt.show()
```

Now we can query the model and get some results!


```
retrieve_images(model,
             image_embeddings,
             query="dogs in park",
             image_filenames=train_df['image'].values)
```



![png](/images/autoencoders_files/autoencoders_66_0.png)




```
retrieve_images(model,
             image_embeddings,
             query="dogs on grass",
             image_filenames=train_df['image'].values)
```



![png](/images/autoencoders_files/autoencoders_67_0.png)




```
retrieve_images(model,
             image_embeddings,
             query="dogs running",
             image_filenames=train_df['image'].values)
```



![png](/images/autoencoders_files/autoencoders_68_0.png)




```
retrieve_images(model,
             image_embeddings,
             query="humans dancing and singing",
             image_filenames=train_df['image'].values)
```



![png](/images/autoencoders_files/autoencoders_69_0.png)




```
retrieve_images(model,
             image_embeddings,
             query="humans doing outdoor activities",
             image_filenames=train_df['image'].values)
```



![png](/images/autoencoders_files/autoencoders_70_0.png)




```
retrieve_images(model,
             image_embeddings,
             query="humans riding bikes",
             image_filenames=train_df['image'].values)
```



![png](/images/autoencoders_files/autoencoders_71_0.png)



# Convert to PDF


```
# generate pdf
# Please provide the full path of the notebook file below
# Important: make sure that your file name does not contain spaces!

# Ex: notebookpath = '/content/drive/My Drive/Colab Notebooks/PS9_javiersc.ipynb'
from google.colab import files
notebookpath = '/content/drive/My Drive/Colab Notebooks/PS9_javiersc.ipynb'

file_name = notebookpath.split('/')[-1]
get_ipython().system("apt update && apt install texlive-xetex texlive-fonts-recommended texlive-generic-recommended")
get_ipython().system("jupyter nbconvert --to PDF {}".format(notebookpath.replace(' ', '\\ ')))
files.download(notebookpath.split('.')[0]+'.pdf')
```

    Hit:1 http://security.ubuntu.com/ubuntu bionic-security InRelease
    Hit:2 http://ppa.launchpad.net/c2d4u.team/c2d4u4.0+/ubuntu bionic InRelease
    Hit:3 http://archive.ubuntu.com/ubuntu bionic InRelease
    Hit:4 http://archive.ubuntu.com/ubuntu bionic-updates InRelease
    Hit:5 https://cloud.r-project.org/bin/linux/ubuntu bionic-cran40/ InRelease
    Hit:6 http://archive.ubuntu.com/ubuntu bionic-backports InRelease
    Hit:7 http://ppa.launchpad.net/cran/libgit2/ubuntu bionic InRelease
    Hit:8 http://ppa.launchpad.net/deadsnakes/ppa/ubuntu bionic InRelease
    Hit:9 http://ppa.launchpad.net/graphics-drivers/ppa/ubuntu bionic InRelease
    Ign:10 https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64  InRelease
    Hit:11 https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64  InRelease
    Hit:12 https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64  Release
    Reading package lists... Done
    Building dependency tree       
    Reading state information... Done
    5 packages can be upgraded. Run 'apt list --upgradable' to see them.
    Reading package lists... Done
    Building dependency tree       
    Reading state information... Done
    texlive-fonts-recommended is already the newest version (2017.20180305-1).
    texlive-generic-recommended is already the newest version (2017.20180305-1).
    texlive-xetex is already the newest version (2017.20180305-1).
    The following package was automatically installed and is no longer required:
      libnvidia-common-460
    Use 'apt autoremove' to remove it.
    0 upgraded, 0 newly installed, 0 to remove and 5 not upgraded.
    [NbConvertApp] Converting notebook /content/drive/My Drive/Colab Notebooks/PS9_javiersc.ipynb to PDF
    [NbConvertApp] Support files will be in PS9_javiersc_files/
    [NbConvertApp] Making directory ./PS9_javiersc_files
    [NbConvertApp] Making directory ./PS9_javiersc_files
    [NbConvertApp] Making directory ./PS9_javiersc_files
    [NbConvertApp] Making directory ./PS9_javiersc_files
    [NbConvertApp] Making directory ./PS9_javiersc_files
    [NbConvertApp] Making directory ./PS9_javiersc_files
    [NbConvertApp] Making directory ./PS9_javiersc_files
    [NbConvertApp] Making directory ./PS9_javiersc_files
    [NbConvertApp] Making directory ./PS9_javiersc_files
    [NbConvertApp] Writing 437915 bytes to ./notebook.tex
    [NbConvertApp] Building PDF
    [NbConvertApp] Running xelatex 3 times: ['xelatex', './notebook.tex', '-quiet']
    [NbConvertApp] Running bibtex 1 time: ['bibtex', './notebook']
    [NbConvertApp] WARNING | bibtex had problems, most likely because there were no citations
    [NbConvertApp] PDF successfully created
    [NbConvertApp] Writing 3673672 bytes to /content/drive/My Drive/Colab Notebooks/PS9_javiersc.pdf



    <IPython.core.display.Javascript object>



    <IPython.core.display.Javascript object>



```

```
