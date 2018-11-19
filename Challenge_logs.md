# Pytorch Challenge Logs

## Day 1: November 10, 2018

Official start of the challenge. Mostly doing some readings on what the challenge format would be. There would be an intro to pytorch. Then there would be the usual, CNNs and RNNs for DL with PyTorch as the framework. Finally, for the final project (which would be the basis of the course) we would be using transfer learning to come up with an image detection and classifier for flowers (orchids by the looks of it).

## Day 2: November 11, 2018

So today, I have started on the initialization for the challenge. For one, I have joined the [Slack Channel](https://pytorchfbchallenge.slack.com/messages/CDB3N8Q7J/convo/CDB3N8Q7J-1541904940.926900/), its already active and there are already some questions posted. For now I think I can help since I have some experience in the Deep Learning Nanodegree. Also, I have installed pytorch and torchvision as pre-requisites to the course.

Right now I am watching the interview with Soumith Chintala, one of the creators of PyTorch, regarding the history and uniqueness of PyTorch from other frameworks. For one thing, its approach was Python first meaning that the python ways we already now and want are applied to the system. It also has a JIT compiler which bridges the known Deep learning frameworks, caffe, tensorflow, torch, etc.,to be able to convert from one framework to another and also to a deployment ready C code for production. In terms of additional features, the PyTorch team is looking into support for Google Colab (for the free GPUs), more interactive notebooks for trainings and examples and also the use of tensorboard for PyTorch.

Right now its 2:23 PM, I have to pause for this session. I have watched the introduction as well as the introduction to pytorch videos so those are done. I have to do other things for next week but I'll be back probably later tonight. Objective for today is to consume the next lesson which is the Introduction to Pytorch (coding) by Mat Leonard (?).

So now, its 9:00PM. Back at it again. For now the idea is to setup Colab for the notebooks. I found [Colab](https://Colab.research.google.com/notebooks/welcome.ipynb#recent=true) and it looks like there is an option for the use of a repository in GitHub. Seems easy. So for now, I'll move over through the Introduction lessons and see what the first lab would be.

So what to expect:

* [x] - Tensors - The data structure of PyTorch
* [ ] - Autograd which is for calculating Gradients in NN training.
* [ ] - Training of an NN using PyTorch.
* [ ] - Use of PyTorch for Transfer Learning for image detection.

So first up was tensors in PyTorch. I thought tensors was some sort of proprietary naming of PyTorch, it was not. Its basically referring to the unit of tensor. So after that we went on to discuss `torch.mm` which is the matrix multiplication equivalent of `np.matmul` in torch. Also there is `torch.sum` which can also be called as a method `.sum()` which obviously sums up the values inside it. One important piece of information that was given in the introduction was the use of memory between numpy and torch. Obviously, pytorch will have compatibility with numpy so anything (an array for example) defined in numpy can be ported to torch via `torch.from_numpy` and vice versa via `.numpy`. In these operations, the memory used for the array are one and the same. Meaning that an operation done in an array that was ported to torch will also be reflected in the version of numpy since they are at the same memory. Also, the transpose operation `.T` is not used in torch. Instead, to match the dimensions of matrix multiplication, we are advised to use `.reshape(a,b)`, `.resize(a,b)` or `.view(a,b)` operation. It is highly advised to make use of `.view(a,b)` than the other two as they do have some issues according to Mat Leonard.

So here is an interesting trick for Colab and Google Drive mount. This should help in uploading those modules or python scripts like unit tests and others to your notebook. It will allow you to read from your google drive input files and others as well. A useful tip, you can run bash commands directly in the notebook via `!` command so `!ls` should output the list of files in your current drive. Which is neat.

```python
# TODO: The code below would start the initialization of your mounting of Google Drive to a Colab notebook.
!apt-get install -y -qq software-properties-common python-software-properties module-init-tools
!add-apt-repository -y ppa:alessandro-strada/ppa 2>&1 > /dev/null
!apt-get update -qq 2>&1 > /dev/null
!apt-get -y install -qq google-drive-ocamlfuse fuse
from google.Colab import auth
auth.authenticate_user()
from oauth2client.client import GoogleCredentials
creds = GoogleCredentials.get_application_default()
import getpass
!google-drive-ocamlfuse -headless -id={creds.client_id} -secret={creds.client_secret} < /dev/null 2>&1 | grep URL
vcode = getpass.getpass()
!echo {vcode} | google-drive-ocamlfuse -headless -id={creds.client_id} -secret={creds.client_secret}

# The code below should mount the google drive to the notebook.

```

## Day 3: November 12, 2018

Connected with some of my classmates in the Udacity PyTorch Scholarship challenge. Interesting to see some diverse background. I have my blockmate in the challenge as well. I connected with someone from the Network Security field trying to apply Deep learning to his field. I have a Process Engineer with a hobby for AI-DL-ML. I also connected with a Machine Learning Lead from exepedia/hotels.com. Its quite a community we have in the challenge.

Right now I am doing the exercises for the introduction to pytorch module.

```python
from torch import nn

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Inputs to hidden layer linear transformation
        self.hidden = nn.Linear(784, 256)
        # Output layer, 10 units - one for each digit
        self.output = nn.Linear(256, 10)
        
        # Define sigmoid activation and softmax output 
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        # Pass the input tensor through each of our operations
        x = self.hidden(x)
        x = self.sigmoid(x)
        x = self.output(x)
        x = self.softmax(x)
        
        return x

model = Network()
model

```

In the code block above, we are creating a new class using the torch. We are calling the nn module and from that our `Network` will inherit its `nn.Module`. It is **mandatory** to inherit the `nn.Module` for our class. This will allow us to create the neural network on torch. For the initialized values we need to define the layers and the transformation we want which in this case is linear, I am thinking we can call Convolution or Recurrent later. Then there are the activation functions `nn.Sigmoid()` and `nn.Softmax(dim=1)`. It is important to take note of one unique property in PyTorch (or torch?) which is broadcasting. The argument `dim=1` for the softmax function is used to indicate which way the softmax is applied which in this case is on the column, setting it to 0 would mean that it is on a row. This is also important on other operations as well like division `A/B` which needs to have a `dim` argument with it. Finally, we can compile our model by calling out the `Network` class and calling model will print out the summary of the model we have created, more like `model.summary()` in Keras.

To go more deeper we use the `torch.nn.functional` module to define our network. It should be similar to the use of `torch.nn` but we are encouraged to use this one.

```python
import torch.nn.functional as F

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        # Inputs to hidden layer linear transformation
        self.hidden = nn.Linear(784, 256)
        # Output layer, 10 units - one for each digit
        self.output = nn.Linear(256, 10)

    def forward(self, x):
        # Hidden layer with sigmoid activation
        x = F.sigmoid(self.hidden(x)) # First layer (hidden)
        # Output layer with softmax activation
        x = F.softmax(self.output(x), dim=1) # Second layer (output)

        return x
```

Below are some unique codes that in my experience only belongs to PyTorch. I may be wrong at this, and Keras or TensorFlow might have the same function but anyway here they are.

```python
# NOTE: Calling out the weight and bias of a layer
print(model.hidden_1.weight)
print(model.hidden_2.bias)
# This would output the tensors of the weights and bias for that model and layer.

# Set biases to all zeros
model.hidden_1.bias.data.fill_(0)
# sample from random normal with standard dev = 0.01
model.hidden_1.weight.data.normal_(std=0.01)
```

So that portion above was all about setting up a neural network in PyTorch. We should now be able to create a network with multiple layers and different activations between them. Up next for this would be the training portion of the neural networks in PyTorch.

## Day 4: November 13, 2018

Now on exercise number 3 of 8: Training Neural Networks. First off [here](https://pytorch.org/docs/stable/index.html) is the link for the documentation of PyTorch. It comes in handy in getting a deeper understanding of what is being discussed in the notebooks. It has more explanation on the process of [autograd](https://pytorch.org/docs/stable/notes/autograd.html#how-autograd-encodes-the-history). I was reading about the autograd feature and its quite intuitive actually and it is what allows PyTorch to be faster, especially in distributed computation (GPU/CUDA). Its whats going to allow computation of gradients during every pass, which in turn would allow us to manually set a layer to not update (for example, transfer learning).

After autograd we move into [Broadcasting semantics](https://pytorch.org/docs/stable/notes/broadcasting.html) for PyTorch. Apparently, its an inheritance of the NumPy broadcasting semantics. It is solving or pointing out dimensionality match or mismatches when we operate tensors (which we will as we progress). One possible way to reduce this error/complication is to understand the basics of Matrix Multiplication, mostly that it is not commutative (orders matter especially in determining the shape of the output).

```python
#NOTE: This is taken from the Docs of PyTorch.
>>> x=torch.empty(5,7,3)
>>> y=torch.empty(5,7,3)
# same shapes are always broadcastable (i.e. the above rules always hold)

>>> x=torch.empty((0,))
>>> y=torch.empty(2,2)
# x and y are not broadcastable, because x does not have at least 1 dimension

# can line up trailing dimensions
>>> x=torch.empty(5,3,4,1)
>>> y=torch.empty(  3,1,1)
# x and y are broadcastable.
# 1st trailing dimension: both have size 1
# 2nd trailing dimension: y has size 1
# 3rd trailing dimension: x size == y size
# 4th trailing dimension: y dimension doesn't exist

# but:
>>> x=torch.empty(5,2,4,1)
>>> y=torch.empty(  3,1,1)
# x and y are not broadcastable, because in the 3rd trailing dimension 2 != 3
```

Now on to the Exercise: Training Neural Networks, the first thing we need in order to start the training is to define our loss function. In PyTorch the list of [available loss functions](https://pytorch.org/docs/stable/nn.html#id50) are in the documentation. Once we have selected our chosen Loss function (dependent really on what we are trying to do), we can then proceed with the autograd function of `forward` and `backward`.

```python
# NOTE: Here we define a simple linear network
# Build a feed-forward network
model = nn.Sequential(nn.Linear(784, 128),
                      nn.ReLU(),
                      nn.Linear(128, 64),
                      nn.ReLU(),
                      nn.Linear(64, 10),
                      nn.LogSoftmax(dim=1))

criterion = nn.NLLLoss() # We define our loss function
images, labels = next(iter(trainloader))
images = images.view(images.shape[0], -1)

logits = model(images)
loss = criterion(logits, labels) # We also define our actual loss calculation
# NOTE: Below we are shown how Torch can calculate the gradients
print('Before backward pass: \n', model[0].weight.grad) # Show the initial contents of the weight at layer 0

loss.backward() # Do one back propagation to update the weights

print('After backward pass: \n', model[0].weight.grad) # Show again the weight at layer 0 to verify that it updated.

```

Now that we have an idea of how to link up Autograd and loss functions, we can now move up on the Optimizers. We can call on the optimizer available in PyTorch via importing `optim` module. The [documentation](https://pytorch.org/docs/stable/optim.html#how-to-use-an-optimizer) on optimizers show how we can create the optimizer. One unique way for optimizer in PyTorch is that we can create an instance where we have the same optimizer but at different learning rates per layer. Imagine training a network using the same Adam optimizer but the hidden layers have a higher/lower learning rate than the actual layer. That would be very useful.

```python
from torch import optim

# Optimizers require the parameters to optimize and a learning rate
optimizer = optim.SGD(model.parameters(), lr=0.01)
```

Moving forward, we already have defined our network, our loss function and our optimizer so how do we train the network? In terms of PyTorch it would mean calling on the `.step()` function in the optimizer. This would tell the network that the loss has been back propagated and the optimizer can now adjust the weights of the networks accordingly to account for the loss.

```python
for input, target in dataset:
    optimizer.zero_grad()
    output = model(input)
    loss = loss_fn(output, target)
    loss.backward()
    optimizer.step()
```

Now that we have an idea of what the model would look like from the boiler plate above, we should be able to do a simple network with it.

```python
## Your solution here
# TODO: Implement a training pass for the network
model = nn.Sequential(nn.Linear(784, 128),
                      nn.ReLU(),
                      nn.Linear(128, 64),
                      nn.ReLU(),
                      nn.Linear(64, 10),
                      nn.LogSoftmax(dim=1))

criterion = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.003)

epochs = 5
for e in range(epochs): # Define how many epochs we would want to train
    running_loss = 0  # Initialize the loss first to zero
    for images, labels in trainloader:
        # Flatten MNIST images into a 784 long vector
        images = images.view(images.shape[0], -1)

        # TODO: Training pass
        optimizer.zero_grad()  # We always initialize this to zero every pass
        logits = model.forward(images)  # We evaluate a forward pass of our model.
        loss = criterion(logits,labels)  # We calculate our loss.
        loss.backward()  # We do one backward pass of it to let the autograd know the loss.
        optimizer.step()  # We call in the step so that our optimizer knows that it can now update the weights.
        running_loss += loss.item()  # We also want to update the loss of our network to know our progress.
    else:
        print(f"Training loss: {running_loss/len(trainloader)}")  # After our model takes a step, we wan to publish its results for verification. We can be more creative on this and plot it later.
```

Now that there is a model that was trained, our weights should now be updated an our model should now be good to go. Below is a sample of a way we can use the trained model for checking of results. As we can see we need to turn off the gradient updates. This is done because we are do not intend to train our network, we just want to try to pass a sample image to it. Without calling on the `no.grad()` function, the model will actually consume memory even if we are not doing any operations.

```python
%matplotlib inline
import helper

images, labels = next(iter(trainloader))

img = images[0].view(1, 784)
# Turn off gradients to speed up this part
with torch.no_grad():  # This will turn of the gradient updates.
    logits = model.forward(img)  # We make a forward pass on our model based on the input image

# Output of the network are logits, need to take softmax for probabilities
ps = F.softmax(logits, dim=1)
helper.view_classify(img.view(1, 28, 28), ps)  # From our helper, we call on the function to help us view the input image and the distribution of the probabilities.
```

In the example above, we were able to code a network in python for identifying an image from the MNIST dataset, handwritten numbers. Also, do note that this was done with a linear neural network. A CNN would arguably do a better job than this and could handle more difficult tasks. In the meantime, we will one up our training and we will be taking on training a neural network to identify from a more complex dataset, Fashion MNIST.

Installing helper function python codes or pull files from a URL to the notebook. Alternatively, there is an option to upload files in the notebook. It is the third tab from the Table of Contents page under `Files`.

```python
#NOTE: Use the code below to install
import os
if not os.path.isfile('helper.py'):

! wget https://GitHub.com/iocfinc/deep-learning-v2-pytorch/blob/master/intro-to-pytorch/helper.py  #Insert the  URL for the file
```

Use the code below to install PyTorch, alternatively just use code snippets.
```python
#NOTE: Use the code below to install PyTorch in Colab
# http://pytorch.org/
from os.path import exists
from wheel.pep425tags import get_abbr_impl, get_impl_ver, get_abi_tag
platform = '{}{}-{}'.format(get_abbr_impl(), get_impl_ver(), get_abi_tag())
cuda_output = !ldconfig -p|grep cudart.so|sed -e 's/.*\.\([0-9]*\)\.\([0-9]*\)$/cu\1\2/'
accelerator = cuda_output[0] if exists('/dev/nvidia0') else 'cpu'

!pip install -q http://download.pytorch.org/whl/{accelerator}/torch-0.4.1-{platform}-linux_x86_64.whl torchvision
import torch
```

## Day 5: November 14, 2018

Currently having some issues downloading the dataset from Fashion MNIST and MNIST via Colab or local. Something about an OSError not reading the correct files from the link. Posted it on the Slack channel and got some responses on how to resolve it. They said to try downloading the file again as it might be corrupted which could explain the issue. As a workaround, I opened up the Python Terminal and ran the code from there. Interesting enough, it was able to download the files for the data set. I am not sure why it was throwing a non-gzip file error when I ran it on the notebook. But at least that is resolved. For now, more exercises on training the neural network for Fashion Mnist data set classification.

With regards to using PyTorch in Colab, [here](https://cloud.google.com/blog/products/ai-machine-learning/introducing-pytorch-across-google-cloud) is a link detailing how this could be achieved (**in the future**) for now, TPUs are only for TensorFlow in Colab which does make sense seeing as they are both Google managed. The good thing is that there is active colaboration between engineers to allow PyTorch the use of TPUs on Colab. Also, this is a [tutorial](https://GitHub.com/nataliasverchkova/on-using-google-Colab) for using Google Colab. Here are some more resources. This one is [about using the GPU in Colab](https://medium.com/deep-learning-turkey/google-Colab-free-gpu-tutorial-e113627b9f5d). Then we have [this one](https://jovianlin.io/pytorch-with-gpu-in-google-Colab/) which is basically a starters guide on Colab and how to use it (brief explanation).

So 2:00 PM right now, my objective is to figure out how to use Colab for this challenge. First off would be mounting Google Drive to the virtual machine. This would be useful in keeping your files at the same place. This also takes care of the dependencies and helper file problems as well as the input files and output files required in running a notebook. To mount it just copy and past the cell below.

```python
# NOTE: Mounting Google Drive to the virtual machine
from google.Colab import drive
drive.mount('/content/gdrive')
```

If done correctly, it should show up in the Files section of the notebook.

<p align="center"><img src='.\Images\GoogleDrive-Mount.png' width=1000px alt = "Mounted Drive"></p>

Now that the drive is mounted, we can now output our files or read our files from it. The code below is from the code snippets of Colab which will create a text file.

```python
with open('/content/gdrive/My Drive/foo.txt', 'w') as f:
  f.write('Hello Google Drive!')
!cat /content/gdrive/My\ Drive/foo.txt
```

<p align="center"><img src='.\Images\Foo.png' width=250px></p>

To upload a file from GitHub to your Google drive we can use `!wget`. For example we would be getting the mnist.py from the Keras GitHub repo.

```python
!wget https://raw.githubusercontent.com/keras-team/keras/master/examples/mnist_cnn.py -P 'gdrive/My Drive/Colab Notebooks'
```

It is important to note somethings when using this method, first is that the file should be in the raw format, otherwise the file will have the html formatting included which would make it unreadable in python. Second is that the location argument should be valid, otherwise a new directory would be created (*save yourself the trouble of debugging due to incorrect spellings*). It should appear both in the Files section of the notebook as well as the Google Drive directory. Or alternatively, you can just download the file itself and upload it manually to your drive. Do note that you can upload your local files to Google Drive as well and the notebook should still see it as long as the Google Drive is mounted. This would be useful if you have some data sets in your local machine.

<p align="center"><img src='.\Images\mnist-upload.png' width=500px></p>

Next is we will install python dependencies. For example, Keras and PyTorch are not pre-installed in the Colab notebooks. To get them we can use `pip install` in Colab notes. You can try to search for the Code Snippets as well if there is already a code that you can add to the notebook to install some dependencies.

```python
!pip install -q keras
```

Actually, Colab is smart enough to figure out what you are trying to do and provide the code snippet for you. For example if we try importing an missing dependency, it would automatically flag it and open up the code snippets.

<p align="center">
 <img src='.\Images\Codesnippet-install.png' width=700px>
</p>

Clicking on the 'insert' link in the code snippet would add the code block to get the module, pytorch in this example. All we have to do is run the code block and that should take care of the installation.

We can also run python `.py` script directly in the notebook. Recall that we downloaded the `mnist_cnn.py` file earlier? It is actually a complete script that would run a CNN network for Mnist. Similar to what we would be doing when we are in a terminal, we just need to call python and point to the file we are going to run. It should be the same thing for other files, just as long as they are visible to the notebook or uploaded to Google Drive.

```python
!python3 "gdrive/My Drive/Colab Notebooks/mnist_cnn.py"
```

<p align="center"><img src='.\Images\Execute py files.png' width=800px></p>

And just to get a glimpse of how powerful Colab is (with GPU of course) just look at the speed at which it went through training the epochs. 14s to 9s for 60000 images. In terms of accuracy, MNIST is fairly easy relative to real world datasets. Its the `Hello World!` of CNN so its understandable that the accuracy can get as high as 99.24%.

<p align="center"><img src='.\Images\mnist-CNN-Run.png' width=800px></p>

`Up next would be cloning a GitHub repo and running those in Colab. We will see if that resolves the helper files issue.`

I was working on figuring out how to clone a GitHub repository to my Google Drive so that we can work from there. As it turns out, I was using the wrong command prefix(?). The general idea is that once the drive is mounted, I need to do changed the directory via `cd`. Once I am in the correct location that is when I will call on `git clone...`. So in short, I need a "scratch" notebook where I can do the sort of terminal commands like cloning and uploading files and fetching data. Once the repositories are uploaded to the drive, that is when I can transfer to those notebooks. Its some sort of nested virtual machines. Its a bit chaotic in the beginning but it makes sense. StackOverflow saves the day for this one. This post regarding [changing the environment](https://stackoverflow.com/questions/48298146/changing-directory-in-google-Colab-breaking-out-of-the-python-interpreter) made it more understandable. Using the `!` command is actually applying the command to the current python interpreter subshell. Using the `%` command changes the current working directory for the notebook environment.

```python
# This will change the working directory, take note of the \ to account for the space character.
%cd gdrive/My\ Drive/Colab\ Notebooks/
# Confirm that you are at the correct directory
!pwd
# use the git clone command to copy the clone to the virtual environment
%git clone <GitHub repo url>
```

Once the repository has been cloned, it should appear now as a folder in your drive. This would allow you to open/edit/save the notebooks via colaboratory. Do note that there is a limit to the free size of Google Drive which is 15Gb. 

<p align="center"><img src='.\Images\Succesful-Clone.png' width=800px></p>

Now that the repository is already in our drive, we can already use colaboratory to open them. So now we can actually make use of the free GPU in Colab to test and train our AI/ML/DL projects. Its also a good thing that Google allows us to use these things for free. Reading up on a background of colaboratory, it was actually similar to TensorFlow which was Google's in-house notebook framework allowing them to work with projects in Data Science and AI internally. Once they had a working version it was then released free to the public together with the infrastructure.

<p align="center"><img src='.\Images\Open-Notebooks.png' width=700px></p>

The free GPU in Colab is also limited both in availability and in use time. If I read correctly the time limit for the session would be 12 Hours, double that of Kaggle (which is also Google owned). In terms of GPU availability, I have not experienced it yet but you will know if there are no available sessions with GPU support because there will be a prompt. Assuming that there are 10,000 scholars in the PyTorch challenge, it would be amazing to see if Colab slows down. Hopefully it will not. That should be all for now at Day 5. We will clone the repo for the Deep learning nanodegree and work on the exercises again tomorrow. Considering its still day 5 I am doing a great pace in the challenge. Still a long way to go but at least there is progress.

## Day 6: November 15, 2018

First off, a continuation of the topic yesterday is the use of cuda cores for PyTorch. After installing PyTorch in Colab we need to first do two things to enable the use of Cuda Cores. First is we set the runtime Hardware Acceleration to a GPU instance. Then the next would be adding the code below to notebook *just before* training the network. Take note that in the `net.cuda()` call, `net` is the name of the model that was defined. If for example we instead used `modlel = Classifier()` then it should be `model.cuda()` to enable the use of the Cuda cores.

```python
use_cuda = True
if use_cuda and torch.cuda.is_available():
  net.cuda()
print(use_cuda and torch.cuda.is_available())
```

One thing that I was not able to solve is how to pull updates from cloned repo to Google Drive. I tried using the `!git fetch` command but that was not working for me. It did run but the file was not reflecting to the Google Drive folder. I had to remove the actual folder in Drive from the front end and update the GitHub repo *before* using `!git clone` again. Its not ideal but its what works for now.

So now that we have been able to setup Colab and clone our repositories to Colab, we can now resume the Exercises for the Intro to PyTorch series. We are now at Part 5 which is about inference and validation. What we are going to do basically is study **overfitting** of a network and use a trained network to predict the outcome. First thing we have to discuss is **overfitting**. This is what happens when our model is actually trying to memorize the features of a data instead of trying to learn the general concept of the data set. The example given here is about studying in an exam. When you overfit, you study past quizzes to the point that you already memorized the answers. The problem here is that you will not be able to adjust when the questions change in the exam because all you did was memorize. You can also be *underfitted* where you skimped through all the possible lessons and topics that might come out. While you will be able to figure out some answers you would not be able to get many of them correctly as you have not made any effort to relate all the facts into one cohesive knowledge. The best fit would then be when you are able to understand a topic in a way that you can explain the concept behind it. In terms of its features, you know how each are connected to one another. The same holds true for our model. We do not want it to memorize the input since it will not do well once we change our inputs from training to testing sets. We also want it to have layered connection so that it can actually make inferences based on the features and context given on each layer. Ideally, our model should be trained such that it is able to identify the features that matter for an object in order for it to relate the different features to come up with a probability distribution for classification.

```python
# http://pytorch.org/
from os.path import exists
from wheel.pep425tags import get_abbr_impl, get_impl_ver, get_abi_tag
platform = '{}{}-{}'.format(get_abbr_impl(), get_impl_ver(), get_abi_tag())
cuda_output = !ldconfig -p|grep cudart.so|sed -e 's/.*\.\([0-9]*\)\.\([0-9]*\)$/cu\1\2/'
accelerator = cuda_output[0] if exists('/dev/nvidia0') else 'cpu'

!pip install -q http://download.pytorch.org/whl/{accelerator}/torch-0.4.1-{platform}-linux_x86_64.whl torchvision

import torch
from torchvision import datasets, transforms

# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
# Download and load the training data
trainset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# Download and load the test data
testset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True, train=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)

import torch
from torch import nn, optim
import torch.nn.functional as F

class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 10)

    def forward(self, x):
        # make sure input tensor is flattened
        x = x.view(x.shape[0], -1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.log_softmax(self.fc4(x), dim=1)

        return x

model = Classifier()

images, labels = next(iter(testloader))
# Get the class probabilities
ps = torch.exp(model(images))
# Make sure the shape is appropriate, we should get 10 class probabilities for 64 examples
print(ps.shape)

top_p, top_class = ps.topk(1, dim=1)
# Look at the most likely classes for the first 10 examples
print(top_class[:10,:])

equals = top_class == labels.view(*top_class.shape)

accuracy = torch.mean(equals.type(torch.FloatTensor))
print(f'Accuracy: {accuracy.item()*100}%')

use_cuda = True
if use_cuda and torch.cuda.is_available():
  model.cuda()
print(use_cuda and torch.cuda.is_available())

model = Classifier()
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 30
steps = 0
a = len(trainloader)
b = len(testloader)
train_losses, test_losses = [], []
for e in range(epochs):
    running_loss = 0
    for images, labels in trainloader:
        # NOTE: This block below is boilerplate.
        optimizer.zero_grad()
        log_ps = model(images)
        loss = criterion(log_ps, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    else:
        ## TODO: Implement the validation pass and print out the validation accuracy
        test_loss,accuracy = 0,0
        with torch.no_grad():
            model.eval()
            for images, labels in testloader:
                output =  model(images)
                test_loss += criterion(output,labels)

                prob = torch.exp(output)
                top_p, top_class = prob.topk(1, dim=1)

                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor))
        model.train()
        train_losses.append(running_loss/a)
        test_losses.append(test_loss/b)
        print('Epoch = {}, train_loss = {:.3f}, test_loss = {:.3f}'.format(e+1, running_loss/a, test_loss/b))
        print('Accuracy: {:.3f}'.format(accuracy/b))
```

The entire code above is the whole code for a vanilla version of Neural Network used in classifying an image from Fashion MNIST into its appropriate category. While the model above works, it has a problem of being over fitted. Below is the graph of the loss between training and validation. As we can see, the training loss constantly decreased but the validation loss did not improve. A brief background between training and validation is that they are separate data used to gauge how the model is progressing during training. To put simply in the context of overfitting, the two are separate sets of inputs. What happens essentially is that the network is overtraining on the training set (due to the repetition) and it simply memorizes the answers instead of the features. This eventually leads to poor performance or no improvement in the scores when the model is shown the different input taken from the validation set.

<p align="center"><img src='.\Images\OverFit.png' width=700px></p>

There is a general explanation to why overfitting happens. At first the weights of the model are randomly generated. At the first pass, here would be nodes that gets triggered so they get updated first. On the second pass, the same neurons may again get activated further increasing the bearing of their magnitude to the outcome of the network. Doing this repeatedly and those nodes that had an early start in training eventually get trained more and as a result gets more "reliable" in a sense that it contributes more to the accuracy. What happened was that the model became specialized into solving just one set of problem.

To help prevent overfitting, we are introduce some randomness to our model in such a way that it discourages overtraining of nodes therefore distributing the weights through out all the possible nodes. One way this is achieved is via the *Dropout* method. The idea behind dropout is that a node will have a chance to be turned off every cycle forcing the network to compensate by updating the weights into other nodes. This way, we are theoretically distributing the weights of our network to better provide accuracy and responsiveness to a different set of inputs. This essentially prevents just some nodes to be trained repeatedly that the model discounts the other nodes' bearing in the model.

To introduce dropout in torch, we use `nn.Dropout`. From the [docs](https://pytorch.org/docs/stable/index.html) we can make use of the dropout layer and wrap our `ReLU` layers with it. With that lets take a look at the modified code.

```python
## TODO: Define your model with dropout added
from torch import nn, optim
import torch.nn.functional as F

class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 10)
        self.dropout = nn.Dropout(p=0.25)

    def forward(self, x):
        # make sure input tensor is flattened
        x = x.view(x.shape[0], -1)

        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.dropout(F.relu(self.fc3(x)))
        x = F.log_softmax(self.fc4(x), dim=1)

        return x
```

Note that the only thing we have to change is how we defined our network. What we added first was the definition of our `self.dropout` layer which in this case has a probability of 0.25 or 25% dropout. One thing to note here is that we can define multiple dropout layers for more customized experience but since this is a fairly small network, we can stop here. After we have initialized our dropout layer, we just have to wrap our original relu-linear layers with the dropout for example `self.dropout(F.relu(self.fc1(x)))`. Before I forget, the code `def forward` is actually required in the case of `torch.nn` this is where our gradients are computed, this is based on the documentation. So back to droup out, now that we have introduced our dropouts to the network we should get an improvement in terms of our validation losses. As seen on the graph below, our validation loss is decreasing together with our training loss although it is still high but that is already an improvement.

<p align="center"><img src='.\Images\Fit.png' width=700px></p>

Some addtional reading can be found in this tutorial from PyTorch regarding [CIFAR10 Classification](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html). It touches briefly on the creation of the network for a classifier. Then it also has additional code to check the total accuracy of the network and how to make use of Cuda cores for the increase in speed.

Okay, some additional instructions for dealing with notebooks that require external files like helper files when using Google Colab.

```python
# TODO: First we need to mount Google Drive again
from google.Colab import drive
drive.mount('/content/gdrive')

# TODO: Next is that we have to change the directory of our environment to point to the correct folder in Google Drive where our files reside.

%cd gdrive/My\ Drive/Colab\ Notebooks/Udacity-Pytorch_Challenge/Exercises
!pwd
!ls

# NOTE: In case you just need 1 File, you can use these instead
# wget
!wget <url>
# Curl
! curl -o <file name> <url>
```

Now that we have trained our model with good accuracy, we can now use it to predict something. In our case its classifying the correct class that the image belongs to.

<p align="center"><img src='.\Images\Result-Inference.png' width=700px></p>

So now we can proceed with saving and loading the network in PyTorch. I am now at Part 6 of 8. This is not an exercise but a tutorial which would be useful for succeeding exercises. To save models, we just use `torch.save`. Based on the [tutorial](https://pytorch.org/tutorials/beginner/saving_loading_models.html), the save module uses Python's pickle utility. To load we just call on `torch.load`. To load the model's parameter, we use `torch.nn.Module.load_state_dict` which would output all the parameters saved in the model's dictionary. So first of all what is a `state_dict`? These are the dictionary mappings for each layer to its corresponding parameter. For the model itself, this would be the weights and biases of every individual layer in that network.

```python
# Saving models
torch.save(<model name>.state_dict(), <path>)
# Loading models
model = modelClass(*args, **kwargs)
model.load_state_dict(torch.load(<path>))
model.eval()  # Call on this BEFORE running any inference when you have dropouts and batch normalization layers.
```

From the tutorial, we get the following recommended methods when saving and loading models. The common convention for saving a model is to use the extension `.pt` or `.pth` for the path filename. Also, it is important to note that `load_state_dict()` requires that our path is deserialized first into its dictionary format, this is the reason for calling `torch.load(<path>)` first.

```python
# Alternative way but not recommended
torch.save(<model name>,<path>)

model = torch.load(<path>)
model.eval()
```

In the alternate way of saving and loading, we are making use directly of pickle. We do not save the mappings of the state but instead save the model class directly. This could potentially lead to issues when used or loaded into other projects.

```python
# General Checkpoint, can be used to load the model for use or when we want to resume training.
torch.save({
    'epoch':epoch,
    'model_state_dict':model.state_dict(),
    'optimizer_state_dict':optimizer.state_dict(),
    'loss':loss,
    <additional arguments>:<additonal argument variables>},
    <path>)

model = modelClass(*args, **kwargs)
optimizer = optimizerClass(*args, **kwargs)

checkpoint = torch.load(<path>)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']
<Additional arguments/variables>

model.eval()
# or 
model.train()
```

The code block above shows how we can customize our saved model. What we can do is we can create a dictionary with the values we want to save. This would be useful for example when we want to pause training and check our progress, or if we want to save checkpoints after some iterations. This could also be useful when we want to share our model to others so that they can continue working on it or in cases of transfer learning. By understanding that the `torch.load()` module requires a serialized dictionary input, we can use that to our advantage by customizing as much of information as we want. To load the model, we just have to input the correct key mapping and we can go from there.

```python
# Warmstarting model using parameters from a different model
# Saving model A
torch.save(modelA.state_dict(),path)
# Creating and loading to model B
modelB = modelBClass(*args, **kwargs)
modelB.load_state_dict(torch.load(<path>), strict = False)
```

The code above is for warmstarting another model from a previous model's parameters (i.e. Transfer Learning). There would be cases where we might want to augment a working model from our colleague and add some more layers or change some parameters to get better results or to match our use case. This is where warmstarting of models are applied. By adding the argument `strict = False`, we are telling the function to ignore keys that are not matching.

In the tutorials for saving and loading there are [more options](https://pytorch.org/tutorials/beginner/saving_loading_models.html#saving-loading-model-across-devices) discussed. Since PyTorch can run on both CPU and GPU, there are methods on which the model is saved to ensure compatibility between those devices, for example saving on CPU and loading on GPU and vice versa. Those are covered in the tutorial in PyTorch.

## Day 7: November 16, 2018

Something is off with the dating, or is it due to the delay between ASIA times and the actual start of the program. Anyway, today is an off day, no coding or lectures done for today. I did some xml parser code for work earlier today. I have been searching in Coursera regarding a possible specialization/course in Data, Big Data and SQL. The idea is that ML and AI or DL is just a small item in the pipeline.

## Day 8: November 17, 2018

I will be resuming the lessons and exercises for PyTorch. After learning to save and load models earlier, we should now be ready to do transfer learning.

I was searching for possible courses to take with regards to the Data-centric lean I want to go to for the next course. I saw this [Data Engineering review](http://www.tribalism.com.au/news/i-completed-data-engineering-on-google-cloud-platform-specialization-coursera-review/) for a specialization in Coursera/Google which is titled [Data Engineering on Google Cloud Platform Specialization](https://www.coursera.org/specializations/gcp-data-machine-learning). An alternate would have to be [Machine Learning with TensorFlow on Google Cloud Platform Specialization](https://www.coursera.org/specializations/machine-learning-tensorflow-gcp). Here is the [medium post](https://medium.com/google-cloud/data-engineering-on-google-cloud-platform-coursera-courses-ca24840d2a3a) from the author of the Data Engineering specialization. In terms of pricing, the fee for Coursera is a monthly 50USD which should provide access to all the Courses and Specializations on their website. That is 3000 a month which is possible for me. One thing that I want to learn from here is where does my AI-ML-DL course stand in terms of the structure of a production ready environment. What I mean is that while AI-ML-DL serves as the core differentiator, it is just a small part of the big picture. By learning Data Engineering and Machine learning on GCP, I am aiming to get a good grasp of what to expect in the prod. Most of the courses in Udacity or the tutorials on the topic already have the dataset cleaned and ready for input, in my opinion this is leading on to a false sense of security in terms of being on the actual field where the data can come from multiple streams and the data is not that clean. Also, there is a free shirt from GCP when we finish by November 30.

[AWS Fundamentals](https://www.coursera.org/learn/aws-fundamentals-going-cloud-native)

## Day 9: November 18, 2018

Its bad but I slept in today. Starting late in Part 8 - Transfer Learning exercises.

## Day 10: November 17, 2018

Early start for the day. Currently doing Part 8. This is about Transfer Learning, where we load parameters of an already trained model. From those parameters, we can warmstart our model to classify our own data which would mean that we are building on top of the model that we have loaded. In the case of the exercise, we are tasked to build a Cat and Dog Classifier. But what is a notebook without a few hiccups. So first error encountered was that there was no data/file in the Folder. Checking on the address it was pointed showed that the file has not yet been downloaded, might have been fixed in the later repo updates but in my case it is not there. So going over to the Slack channel I found the link for dataset and used wget to download it to my Google Drive. The file was ~536Mb but luckily this is a cloud service so Google's connection took care of the size. The unpacking was the long part since this is a whole set.

```python
%cd <Active directory> # Make sure to change your directory first

!wget https://s3.amazonaws.com/content.udacity-data.com/nd089/Cat_Dog_data.zip
!unzip Cat_Dog_data.zip
```

Since we now have the data set downloaded, I just have to recheck the directory the loadfile is located and make sure its available After those checks we then moved through normalization. Recall from Part 7 that when loading a model, we have to match some of our parameters to the ones used in the model to be loaded. In our case we have to define the correct normalization parameters for our model as well. The different color channels for the images where normalized using different means and standard deviations.

```python
# For normalizing in our transformations
transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
# For the entire train_transform

train.transforms = transforms.Compose([transforms.RandomRotation(30),
                                       trasnforms.RandomResizeCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485,0.456,0.406],
                                                            [0.229,0.224,0.225])])
```

In the case of our training data, we have several transforms to do. One is `RandomRotation` which will rotate to a certain degree our images. Second is we have `RandomResizeCrop` which will crop our images to a certain size while also randomly selecting which portion to crop off. We then have `RandomHorizontalFlip` which is exactly what is says in its name, flip the images in the horizontal axis. We have transform `ToTensor` which will make our array as tensors for use by pytorch. Finally, we have `Normalize` which will evaluate our tensor and match it to the normalization parameters we have set. Do note that the functions we have added to our train composition are there for the purpose of dataset augmentation. One reason for doing this is so that we can add some resiliency to our trained network. It is not always the case that an image is always centered or the image is on the correct orientation. It also happens sometimes that there are not enough samples in our data set that we have to add/augment some of it by randomly "manipulating" our original set.

* [x] - Tensors - The data structure of PyTorch
* [x] - Autograd which is for calculating Gradients in NN training.
* [x] - Training of an NN using PyTorch.
* [ ] - Use of PyTorch for Transfer Learning for image detection.
* [x] - Figure out using Colab for the challenge. There is a GPU and TPU service on the cloud. :joy:
