# PyTorch Challenge Logs

## Day 1: November 10, 2018

Official start of the challenge. Mostly doing some readings on what the challenge format would be. There would be an intro to PyTorch. Then there would be the usual, CNNs and RNNs for DL with PyTorch as the framework. Finally, for the final project (which would be the basis of the course) we would be using transfer learning to come up with an image detection and classifier for flowers (orchids by the looks of it).

## Day 2: November 11, 2018

So today, I have started on the initialization for the challenge. For one, I have joined the [Slack Channel](https://pytorchfbchallenge.slack.com/messages/CDB3N8Q7J/convo/CDB3N8Q7J-1541904940.926900/), its already active and there are already some questions posted. For now I think I can help since I have some experience in the Deep Learning Nanodegree. Also, I have installed PyTorch and torchvision as pre-requisites to the course.

Right now I am watching the interview with Soumith Chintala, one of the creators of PyTorch, regarding the history and uniqueness of PyTorch from other frameworks. For one thing, its approach was Python first meaning that the python ways we already now and want are applied to the system. It also has a JIT compiler which bridges the known Deep learning frameworks, caffe, tensorflow, torch, etc.,to be able to convert from one framework to another and also to a deployment ready C code for production. In terms of additional features, the PyTorch team is looking into support for Google Colab (for the free GPUs), more interactive notebooks for trainings and examples and also the use of tensorboard for PyTorch.

Right now its 2:23 PM, I have to pause for this session. I have watched the introduction as well as the introduction to PyTorch videos so those are done. I have to do other things for next week but I'll be back probably later tonight. Objective for today is to consume the next lesson which is the Introduction to PyTorch (coding) by Mat Leonard (?).

So now, its 9:00PM. Back at it again. For now the idea is to setup Colab for the notebooks. I found [Colab](https://Colab.research.google.com/notebooks/welcome.ipynb#recent=true) and it looks like there is an option for the use of a repository in GitHub. Seems easy. So for now, I'll move over through the Introduction lessons and see what the first lab would be.

So what to expect:

* [x] - Tensors - The data structure of PyTorch
* [ ] - Autograd which is for calculating Gradients in NN training.
* [ ] - Training of an NN using PyTorch.
* [ ] - Use of PyTorch for Transfer Learning for image detection.

So first up was tensors in PyTorch. I thought tensors was some sort of proprietary naming of PyTorch, it was not. Its basically referring to the unit of tensor. So after that we went on to discuss `torch.mm` which is the matrix multiplication equivalent of `np.matmul` in torch. Also there is `torch.sum` which can also be called as a method `.sum()` which obviously sums up the values inside it. One important piece of information that was given in the introduction was the use of memory between numpy and torch. Obviously, PyTorch will have compatibility with numpy so anything (an array for example) defined in numpy can be ported to torch via `torch.from_numpy` and vice versa via `.numpy`. In these operations, the memory used for the array are one and the same. Meaning that an operation done in an array that was ported to torch will also be reflected in the version of numpy since they are at the same memory. Also, the transpose operation `.T` is not used in torch. Instead, to match the dimensions of matrix multiplication, we are advised to use `.reshape(a,b)`, `.resize(a,b)` or `.view(a,b)` operation. It is highly advised to make use of `.view(a,b)` than the other two as they do have some issues according to Mat Leonard.

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

Right now I am doing the exercises for the introduction to PyTorch module.

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

Now on exercise number 3 of 8: Training Neural Networks. First off [here](https://PyTorch.org/docs/stable/index.html) is the link for the documentation of PyTorch. It comes in handy in getting a deeper understanding of what is being discussed in the notebooks. It has more explanation on the process of [autograd](https://pytorch.org/docs/stable/notes/autograd.html#how-autograd-encodes-the-history). I was reading about the autograd feature and its quite intuitive actually and it is what allows * to be faster, especially in distributed computation (GPU/CUDA). Its whats going to allow computation of gradients during every pass, which in turn would allow us to manually set a layer to not update (for example, transfer learning).

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

## Day 10: November 19, 2018

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

## Day 11: November 20, 2018

No other progress done yesterday. The connection is unstable that the session with Google Colab constantly gets disconnected. So I just went on to read Thinking Fast and Slow. Its unrelated to the topic but it does have its application.

What? A user has already started figuring out the lab project? Josh Bernhard had [this medium post](https://medium.com/@josh_2774/deep-learning-with-pytorch-9574e74d17ad) detailing his project. Its a 20 minute read so I expect it to be bloody.

Anyway back to Part 8 of the PyTorch intro. Its funny when you notice that you are doing something wrong 2 hours into doing it. Talk about a waste of time. I was actually training my loaded model in CPU. No wonder it was taking too long.

```python
model = models.densenet121(Pretrained=True)
model
```

The code above is what we are using to load out a model that is already available in PyTorch. For more models we can refer to this [link] to the PyTorch model zoo. In Udacity's example, they are loading up the densenet121 model. Looking at the zoo, PyTorch has selections that include ResNet, Densenet, VGG, Inception and SqueezeNet with varying depth. I took a screen shot of the results for the networks.

<p align="center"><img src='.\Images\torchvision.models — PyTorch master documentation.png' width=450px></p>

Now that we have loaded up our model, we need to freeze our model's parameters. Specifically, we set our `requires_grad` flag to `False`. This is done because we want to use the pre-trained parameters of our model and build on top of it. We do not want to, at least for the sake of the example, train the weights. We take note that we are using the model to get the features from our image inputs. The model is not yet complete in a sense that it is not yet able to map out the features to the correct label. It still does not have the classifier portion. So we add a few layers at the output of the model we loaded. Some key points to look at here is that the input size of the classifier should be equal to the output of the last layer of the model, in this case 1024 for DenseNet151.

```python
# Freeze parameters so we don't backprop through them
for param in model.parameters():
    param.requires_grad = False

from collections import OrderedDict
classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(1024, 500)),
                          ('relu', nn.ReLU()),
                          ('fc2', nn.Linear(500, 2)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))

model.classifier = classifier
```

Once we have loaded the model, we can already proceed with using it to classify our images. A very important discussion should be in clearing up what happens next and why Deep Learning (especially CNNs) are so difficult to get into. It does not mean that since we are not training, we can get away without the use of a GPU. Our images still have to pass through the model. Seeing DenseNet151's depth should already give us an idea of just how much computation is required for us to be able to get a result. This is where the use of the Cuda cores in GPU or in the case of TensorFlow, TPUs, come in handy. At our scale, several sample images of cats and dogs, we can still get away with using the free GPU in the Google Colab cloud. Introducing ML-AI-DL to a production environment with multiple sources of data, or building a model to classify images requires some serious compute power. Right now this is an important distinction to be pointed out. There is a gap in what we do here in our lab and what really happens out there in production. Just something to think about.

Now that we are clear that we *NEED* to use GPUs, we do some modifications to our code to allow us the use of CUDA. First we need to `import time` so that we can monitor the baseline of how much improvement we have. Then we make a check between the run times in CPU and CUDA devices. To tell PyTorch which device to load the model, we simply call the `.to(<device>)` to our model. In this case its `model.to(<['cpu','gpu']>)`. Do note that when we get new tensors in our computation, we also have to  tell them which device they are to be processed. So we need our `inputs` and `labels` to be processed in the device we are on.

```python
import time

for device in ['cpu','cuda']:
    criterion = nn.NLLLosss()  # Used since we are using softmax in our final output
    optimizer = optim.Adam(model.classifier.parameters(),lr=0.001)
    model.to(device) # Tell PyTorch which device to use, CPU or GPU
    for ii, inputs, labels in enumerate(trainloader):
        inputs, labels = inputs.to(device), labels.to(device)

        start = time.time()  # Record the start time for computation

        outputs = model.forward(inputs)
        loss = criterion(outputs,labels)
        loss.backward()
        optimizer.step()

        if ii==3:  # In here we simply define our batch size as 3
            break
    print(f"Device: {device}; Time per Batch: {(time.time()-start)/3:.3f} seconds")
```

The results are night and day. Using cuda device allows us to compute a batch faster. The reason for this is how computations are made in Deep Learning. Gradients are based on Matrix Operations and each matrix operation can be done in parallel with each other. The bigger the model is, the more gradients it has to compute. The difference between CPU and GPU is that GPU is made for computing small items in parallel while the CPU is designed to take on more intensive computations in sequence. My guess is that throughput is better in our case that speed.

<p align="center"><img src='.\Images\GPU-Vs-CPU.png' height=100px></p>

```python
# Checker of CUDA or CPU device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
```

The code above is a boiler plate for checking if there is a cuda device available in our instance and select it if available, otherwise it would simply default to the CPU.

```python
# NOTE: This is a copy of the bottom portion of DenseNet121

      (denselayer15): _DenseLayer(
        (norm1): BatchNorm2d(960, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu1): ReLU(inplace)
        (conv1): Conv2d(960, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (norm2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu2): ReLU(inplace)
        (conv2): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      )
      (denselayer16): _DenseLayer(
        (norm1): BatchNorm2d(992, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu1): ReLU(inplace)
        (conv1): Conv2d(992, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (norm2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu2): ReLU(inplace)
        (conv2): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      )
    )
    (norm5): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  )
  (classifier): Linear(in_features=1024, out_features=1000, bias=True)
)
```

The code above shows the last layers of DenseNet121 which we will be loading. What we are looking for is the `classifier` layer, specifically the input to that layer. What we will be doing is changing that layer to a sequential one. In this case we have an input of 1024, and we will need to output 2 (cat or dog). The 1000 `out_features` in the model is due to the original data it was trained on which is ImageNet which is a collection of images from 1000 labels ranging from animals like cats and dogs to other things like cars, airplanes etc.

Now that we have an idea of the `in_features` we can recreate the classifier to make it more suitable to our application.

```python
# TODO: Freeze the parameters

from parameters in model.parameters():
    prameters.requires_grad = False

classifier = nn.Sequential(nn.Linear(1024,500),
                            nn.Dropout(0.15),
                            nn.Linear(250,2),
                            nn.Dropout(0.15),
                            nn.Linear(259,2),
                            nn.LogSoftmax(dim=1)
                            )
model.classifier = classifier
criterion = nn.NLLLoss
optimizer = optim.Adam(model.classifier.parameters(),lr=0.0025)
model.to(device)
```

Now that we have modified the classifier layer, we can start passing though our inputs and check our model's accuracy.

```python
epochs = 1 # We just want to check the results, we do not want to train the model.

# Initialization
steps = 0
running_loss = 0 
print_every = 5  # Let's minimize the outputs

for epoch in range(epochs): # Just one pass
  for inputs,labels in trainloader:
    steps+=1  # Increment the step
    inputs, labels = inputs.to(device), labels.to(device) # Pass the inputs to device first for use of Cuda cores.
    #Boiler plate
    optimizer.zero_grad() # Zero out our gradients
    out = model.forward(inputs)
    loss =  criterion(out,labels)
    loss.backward()
    optimizer.step()
    running_loss +=loss.item()

    if steps % print_every ==0:
      # Initializations for test
      test_loss=0
      accuracy = 0
      model.eval()
      with torch.no_grad():
        for inputs, labels in testloader:
          inputs, labels = inputs.to(device), labels.to(device) # Pass the inputs to device first for use of Cuda cores.
          output = model.forward(inputs)
          batch_loss =  criterion(output,labels)
          test_loss +=batch_loss.item()
          # NOTE: No auto-grad zeroing in testing since we are not going to call model.eval()

          # Accuracy of the model
          # Another boiler plate if your think of it
          ps = torch.exp(out)
          top_p, top_class = ps.topk(1,dim=1)
          equals = top_class == labels.view(*top_class.shape)
          accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

      print(f"Epoch {epoch+1}/{epochs}..."
            f"Train Loss: {running_loss/print_every:.3f}..."
            f"Test Loss: {test_loss/len(testloader):.3f}..."
            f"Accuracy: {accuracy/len(testloader):.3f}"
           )
      running_loss=0
      model.train()
```

* [x] - Tensors - The data structure of PyTorch
* [x] - Autograd which is for calculating Gradients in NN training.
* [x] - Training of an NN using PyTorch.
* [x] - Use of PyTorch for Transfer Learning for image detection.
* [x] - Figure out using Colab for the challenge. There is a GPU and TPU service on the cloud. :muscle:

We have just finished the Introduction To PyTorch module of the challenge. We have learned about autograd and tensors and we have used `torch.nn` to create our models. We also were able to save and load our models in PyTorch. Finally, we were able to use transfer learning to make use of existing pre-trained models and modifying them to our application.

Up next would be another discussion on Convolutional Neural Networks. This is going to be the project we need to submit for consideration to the scholarship so we need to listen and take notes well.

With regards to the error, I think it has something to do with defining the classifier or the linking between Resnet that I was trying. I was searching and found this peice of code `test_embeds = Variable(torch,randn(5. 10), requires_grad=True)` which could be worth a try.

## Day 12: November 21, 2018

A bit of a redo of transfer learning for PyTorch. It is very important that I grasp how to do it. What I am doing is using transfer learning again but this time I am loading up Resnet101 instead of DenseNet121 which was used in the example.

Here is a [tutorial](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html) from the PyTorch site. It is taking on FineTuning where we further train the loaded model to our inputs and also Fixed Feature Extraction where we simply build on top of it.

In the tutorials I learned about the autograd functionality of PyTorch. When using a model, recall that we use:

```python
for param in models.parameters():
    param.requires_grad = False
```

We know that this is for freezing up the gradients of the models. I just learned that this is actually only used in Feature Extraction uses of transfer learning. So why is it then that we freeze the gradients when we are going to add our classifier network later? This is answered by the unique default setting of autograd. Newly created modules, even on a model that has frozen gradients, will have their `requires_grad` flag set to True. This is why we can still train our classifier even with the original model's parameters frozen. I think this is similar to the setting in TensorFlow of `is_training`. I know there is an option in TensorFlow to stop updates to the pre-trained model as well selectively choose which layers to freeze.

Now back to transfer learning, I figured out how to add (correctly) to the model's classifier. Its not simply creating a sequential module and calling it classifier. We actually have to take a look at the last layer of the loaded model and link our sequential model to that one.

```python
# TODO: Insert the sequential model here for Feature Extraction

```

I have a problem. I am now lost at what loss model to use. I am trying to use `NLLLoss` for the model as it has a `LogSoftmax` activation for probability in the final layer but I am getting only 60% accuracy. I think I made an error in choosing the criterion. I have to fix this. :confused:

Okay, since I need some refresher for the loss functions and optimizations I checked and found this [article in Medium](https://medium.com/data-science-group-iitr/loss-functions-and-optimization-algorithms-demystified-bb92daff331c) detailing the loss functions and optimizations demystified.

First up is *Error and Loss functions*. In most networks, error is simply the difference between the computed value and the actual value. The function on which the error is computed is called the **loss function**. Different loss functions will have different effects on the performance of the model. The use of loss function is generally dictated by the type of task that the model is used for (regression or classification).

Once we are able to calculate our error via our loss function, we have to update our weights so that the error we get is minimized. This is where our **optimization function** comes in. Optimization functions usually calculate the gradients, which are the partial derivative of the loss function w.r.t. the weights, usually to the opposite direction of the gradient(thus gradient descent).

>> The components of a neural network, the activation function, loss function and optimization algorithm used, play an important role in efficiently calculating and effective training of a model to provide accurate results. Different tasks require a different set of functions to give the optimum results.

Loss functions as stated earlier are used for different tasks. Mainly we have **Regressive Loss functions** where we are predicting a target variable which is continuous. Examples of regressive loss functions are *Mean Squared Error (MSE)*, *Absolute Error* and *Smooth Absolute Error*. Next up would be **Classification Loss functions** where our target is a probability value (score). The target variable is usually binary (True/False), (Cat/Dog), (Ant/Bee) etc. or it can be multi-class. In terms of loss, what it computed is usually the *margin*, which is the measure of how correct we are and how confident our predictions are. Most classification loss would aim to maximize margin. Examples of this would be *Binary Cross Entropy*, *Negative Log Likelihood*, *Margin Classifier* and *Soft Margin Classifier*. Then we have **Embedding Loss functions** which is a measure of similarity between two inputs. Examples of this would be *L1 Hinge Error* that calculates the distance L1 between the two inputs and *Cosine Error* which calculates the Cosine distance between the inputs.

Once we have a loss function we can then proceed with how we minimize losses and optimize our model. This is where optimization comes in. *Optimization functions* dictate how our model gets trained and how fast our convergence would be (minimized/high accuracy model). Some examples of this would be *Adam*, *Stochastic Gradient Descent(SGD)* and *Adagrad*. Most of these are usually a matter of convergence and speed. My main issue right now is to first fix the loss functions so that the optimization can do its part.

In the tutorial example, he gave us the loss function as `CrossEntropyLoss()`. I am not sure if its the same as *Categorical Cross Entropy* but it makes sense that it is entropy based since this is a classification problem. What I am not sure of is why it does not seem to work when I use two linear layers. Also, he is using `SGD` for the optimization while I use `Adam`. In terms of learning, Adam does work in a sense that the accuracy increases. The issue I have is that 
the loss and accuracy are just in the 50-60% range.:disappointed:

```python
model_ft = models.resnet18(pretrained=True)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 2)

model_ft = model_ft.to(device)

criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
```

There is still another [tutorial for finetuning](https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html) which could also provide other answers on the question of the low accuracy. It could just be a really big jump of nodes or there really is a need to add another hidden and that we can simply go from 2048 to 2 directly. :unamused:

More time to do this later and tomorrow but this is very important since this is the basis of getting accepted or not to the scholarship.

Another worthy read would be [Dealing with datasets imbalance](https://towardsdatascience.com/handling-imbalanced-datasets-in-deep-learning-f48407a0e758). Based on the Lab Challenge channel in slack, there is already a discussion on how the dataset is unbalanced. To provide an edge on competitive challenges (like this scholarship) I found a [post on staying on the top 2% of a Kaggle Competition](https://towardsdatascience.com/my-secret-sauce-to-be-in-top-2-of-a-kaggle-competition-57cff0677d3c?source=placement_card_footer_grid---------2-41). We have our plate full of learnings and task. :muscle:

## Day 13: November 22, 2018

Transfer learning fixing classification loss function.

Consumed by work today. :warning:

Made no progress except for reading some medium posts on Loss functions.

## Day 14: November 23, 2018

Another day. Would be trying to catch up today on the lessons. I am now on Convolutional Neural Networks. The instructor is Alex Cook together with another one (I have not yet seen her and know her name).

First of is *Features*. In the context of CNNs they are actually used to find features in the image and from those features provide information or prediction as to what the image is. Features are usually specific for every CNN node, for example one node might be looking for edges and another one is looking for borders and another one is looking for a shade. Combining these features will allow the network to provide a classification as to what it is actually looking at or the input. Another way to thinking about it is about how we humans see an image and classify it. In our case we are able to more specifically check what features to look at. For example we look at the face of a person and check its size, the color, the eyes, the nose, the eyebrows, if there is a mole somewhere. Obviously, for us humans, its almost trivial but that is actually how it is logically how it works if we break it down. Its just "human nature" so we tend to overlook it.

## Day 15: November 24, 2018

Continuing on Convolutional Neural Networks today. Aside from looking at how CNN works, we again retouch on the use of a validation set. Generally, a validation set serves two purpose:

* Measure how well the model is generalizing, during training - It will show us how our model at a certain epoch comes performs to a data that it was not trained on. This would indicate that our model has been able to focus on getting features that matter and classify inputs according to these features.

* Validation loss will also tell us when to stop training the model - Building up on the first purpose of a validation loss, we would want to be able to know when our model is generalizing and at what point is it simply memorizing details from the data. When our validation loss stops improving, or when it starts to increase then we know that the training of our model has to stop. This is because we have indication that the model is simply memorizing the training data, causing it to have a bad performance when used on a validation data set.

Why use CNN instead of MLP? Especially images, CNN would retain the spatial features of the image which cannot be done by MLPs. When talking about detecting features, this is done by applying various filters to an image. These filters are made in such a way that they are able to detect spatial features in the image. When we say *spatial features* we are talking about either *color* or *shapes*. We will mostly deal with *shapes* in image detection. *Shapes* can be thought of as intensity changes, or edges. When we look at something with our eyes, the first thing we look out for is the shape or the overall outline. The same thing is true when we want to look at an image. We first need to know if there is actually a shape in the input image. To do this, we need to have a filter that can detect abrupt changes in intensity which usually indicates an edge. Pass a filter to an image that detects an edge as it convolves and you get to see an outline of the shape. To detect an intensity in an image, we will be creating a specific image filter that looks at a certain group of pixels(filter size) and react to alternating patterns of dark/light pixels (edges). The output of moving around this filter to the entire image is a new image that shows edges of objects and differing textures.

<p align="center"><img src='.\Images\screen-shot-2018-09-24-at-3.18.33-pm.png' height=400px></p>

We are introduced to the concept of frequency in an image. Frequency is just the number of oscillations per unit time of an object. So basically its simply *rate of change*. In the context of an image, frequency is actually the rate of change of intensity for a given area/space. What are the implications of frequency of an image? For us in image detection, its about determining the edges of an object in the image and determining if its a shape or if its a background. In the example above, the blue box shows an area with low frequency. This area of low frequency generally have low variations between pixel values suggesting uniformity and lack of edges. The pink box on the other hand is showing an area of high frequency. The intensity changes between the black and white pixels are discernable. This suggest the presence of edges and possibly shapes in that are which in this case indicates stripes in the area.

Now that we have the idea of the frequency in the image, we move on to the concept of *High-pass filters* in the context of convolution. We generally use High-pass filters because we have learned earlier that the rapid changes in image frequency indicates an edge. From this we create a *Kernel* which serves as the weight on which our image is convolved. We go over the entire image using our kernel and from the convolutions we get a new (processed) value which reflects the result of the convolution of the kernel and the input. In this specific case, we should see an image with the outline of the object in the input image being highlighted.

How do we actually handle edges using kernels? The answer is in creating a kernel with the focus on the center and looking at the neighboring pixels. One important concept in edge detection kernels is that the sum of the elements inside the kernel is 0. This is done because we just want the edges to be detected. Having a sum (bias) of any other number is no longer an edge detection kernel. It is also important how the element values are distributed. Usually, the center element in the 3x3 kernel holds a high value (the focus point). Then the adjacent elements of the center pixel all have a value that when added would be equal to the negative of the value in the center element. For example, the center element is valued 4. This would make the adjacent elements (above, below, left and right) of the center pixel have a value of (-4)/4 or -1. The elements diagonal to the center pixel are usually left at zero as they are the farthest and contribute the least to the edge so it is generally not considered (i.e. element value is multiplied by 0).

How do we deal with the edges of the input (pun intended)? What we are trying to discuss is how do we fit the kernel at the borders of the input image. The kernel is made in a way that the focus pixel is at the central element. So pixels at the edges of the input images will not have a convolution output. To deal with this, we use the concept of padding. This is where the input images are generally padded with zeros so that the kernel will still get applied and the shape of the output will retain the shape of the input while also allowing the convolution to happen. An alternative to padding would have to be cropping the image. This would make the output image a different size to that of the input and that would lead to some pixels not being processed and that constitutes a loss of data/detail which we obviously want to avoid.

Below is an example of how a kernel can be applied to an image with the use of OpenCV library. Here we want to show how images are transformed via the use of a filter we have set.

```python
# Using OpenCV to apply filters
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import cv2
import numpy as np

%matplotlib inline

# Read in the image
image = mpimg.imread('images/curved_lane.jpg')
image_2 = mpimg.imread('images/bridge_trees_example.jpg')

plt.imshow(image)
```

So the output of the code block above would be the image below which just shows the original image we use as input.

<p align="center"><img src='.\Images\curved_road-input.png' height=300px></p>

```python
# Converting to Grayscale
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

plt.imshow(gray, cmap='gray')
```

After importing the image, we usually split it up into its corresponding channels (R,G,B) or in this case convert it to a single channel via gray scaling.

<p align="center"><img src='.\Images\image_grayscaled.png' height=300px></p>

```python
# Creating a custom Kernel
# Create a custom kernel

# 3x3 array for edge detection
sobel_y = np.array([[ -1, -2, -1], 
                   [ 0, 0, 0], 
                   [ 1, 2, 1]])

## TODO: Create and apply a Sobel x operator


# Filter the image using filter2D, which has inputs: (grayscale image, bit-depth, kernel)  
filtered_image_y = cv2.filter2D(gray, -1, sobel_y)

plt.imshow(filtered_image_y, cmap='gray')

```

In the code above the kernel we have defined is in the numpy array named `sobel_y`. What this kernel would do is detect changes on the horizontal edges. The resulting image shows an outline of the detected edges. We can see that it was able to detect high contrasting boundaries like the lane markings and the edge of the concrete barrier in the curve. This should give us an idea of how else we can make use of convolution and kernels  in detecting other features in the image.

<p align="center"><img src='.\Images\image_filtered.png' height=300px></p>

The importance of filters cannot underestimated especially in the application for Convolutional Neural Networks. In the structure of a CNN model, Convolutional layers are generally what track the spatial information and *learn* to extract features like the edges of the objects. The convolutional layer is actually the process and the output. So when we refer to convolutional layers, what we are actually referring to are the output images of applying different convolutional kernels to our input. We have already discussed the kernels which are the matrices that we apply through out input to produce a processed output. We also know that varying the values of the elements inside the kernel allows us to change what our models detect. In the previous example, we have touched on applying kernel filters to an image by defining the kernel values. Since this is deep learning, we can actually make think of the kernel elements as our weights. With this idea of kernels as weights, we can therefore apply the concept of deep learning where we train our model to define its own set of weights for the kernel so that it can detect images and objects in any input image. For example, we can train our model to detect cats or dogs anywhere on the image or for example find a horizon in any image.

<p align="center"><img src='.\Images\conv_layer.gif' height=500px></p>

We then have to discuss some parameters that are important for the use of Convolutional Layers. First is the **stride** which dictates how much our kernel gets moved from its original point to its next point. Stride will affect how much data from our original image gets processed and which ones will contribute to the output. Now that we covered stride, let us move on to the other parameter which is directly linked to stride. This parameter is called **padding**. Padding will tell the Convolutional layer what to do on pixels that do not fit the kernel size we have indicated. Again, this happens on the edges of the input image where the kernel will move past the original dimensions of the image. We can either discard the pixels which are no longer fitting our kernel and input image dimensions or we can use zero padding. In the former method, all the edges will simply get a value of zero as we cannot perform convolution on them due to dimensionality issues. The later option artificially adds zeros to the edges of our input image, enough to allow us to still perform convolution without the loss of the data.

Now that we have covered convolution and its parameters stride and pooling we can proceed with the next layer on our model which is the **Pooling layer**. Pooling layer allows us to reduce the dimensionality of our model therefore reducing the number of parameters which would lead to over fitting if left unchecked. There are two major types of pooling which are used in CNN architectures, the first is **Max pooling** and the second one is **Average pooling**. In the former, the elements in the window size are pooled together and the maximum value among the elements gets to represent that layer. In the later, the elements in the window size are averaged and this becomes the representative value of that layer.

While we have mentioned two pooling options, what is usually used in terms of image classification and object detection is **_Max Pooling_**. The reason for this is that we want the most important details to get represented in our output. Doing an averaging operation would lead to smoothing out of the output which is actually the opposite of what we are trying to achieve.

<p align="center"><img src='.\Images\maxpooling_ex.png' height=200px></p>

Now that we know what the different layers for our CNN model,  we can proceed with its application in code.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
    
# define a neural network with a single convolutional layer with four filters
class Net(nn.Module):

    def __init__(self, weight):
        super(Net, self).__init__()
        # initializes the weights of the convolutional layer to be the weights of the 4 defined filters
        k_height, k_width = weight.shape[2:]
        # assumes there are 4 grayscale filters
        self.conv = nn.Conv2d(1, 4, kernel_size=(k_height, k_width), bias=False)
        self.conv.weight = torch.nn.Parameter(weight)

    def forward(self, x):
        # calculates the output of a convolutional layer
        # pre- and post-activation
        conv_x = self.conv(x)
        activated_x = F.relu(conv_x)

        # returns both layers
        return conv_x, activated_x

# instantiate the model and set the weights
weight = torch.from_numpy(filters).unsqueeze(1).type(torch.FloatTensor)
model = Net(weight)

# print out the layer in the network
print(model)
```

In the code block above, we have defined our convolution in the `__init__` block. We have also defined the characteristics of our `forward` pass. One additional thing we did here is that we loaded up our filters which we have defined our self. Although it is not shown here, it was discussed in the original notebook. To give you an idea of the filters we had, I have added the image below for reference.

Now, the input of the image is a car which is Udacity's self driving car.

<p align="center"><img src='.\Images\input_initialized_pytorch.png' height=500px></p>

When we apply our pre-defined filter to the input image, we get interesting results at the output end. From `Filter_1` we get emphasis on the left edges. From `Filter_2` we get right edge emphasis. The same ideas go to filters 3 and 4 which emphasizes the top-to-bottom transition edges and vice-versa. Once we are able to visualize the output of applying filters to an input, we should get the general idea that stacking multiple filters like these to our input image allows us to detect unique shapes and object outlines.

<p align="center"><img src='.\Images\filters_pytorch.png' width=1000px></p>
<p align="center"><img src='.\Images\Outputs_filtered.png' width=1000px></p>

A final example in the visualization notebook is the use of activation layers for CNNs. The results are shown below. Now that I see these images it looks like the explanation I did earlier was reversed, filter 1 actually shows right edges.

<p align="center"><img src='.\Images\Relu_Ativated_Output.png' width=1000px></p>

## Day 16: November 25, 2018

The examples yesterday were on using CNNs with `ReLU` activations, for our first example today we will be applying `Max Pooling` to our convolutions.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# define a neural network with a convolutional layer with four filters
# AND a pooling layer of size (2, 2)
class Net(nn.Module):

    def __init__(self, weight):
        super(Net, self).__init__()
        # initializes the weights of the convolutional layer to be the weights of the 4 defined filters
        k_height, k_width = weight.shape[2:]
        # assumes there are 4 grayscale filters
        self.conv = nn.Conv2d(1, 4, kernel_size=(k_height, k_width), bias=False)
        self.conv.weight = torch.nn.Parameter(weight)
        # define a pooling layer
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        # calculates the output of a convolutional layer
        # pre- and post-activation
        conv_x = self.conv(x)
        activated_x = F.relu(conv_x)

        # applies pooling layer
        pooled_x = self.pool(activated_x)

        # returns all layers
        return conv_x, activated_x, pooled_x

# instantiate the model and set the weights
weight = torch.from_numpy(filters).unsqueeze(1).type(torch.FloatTensor)
model = Net(weight)

# print out the layer in the network
print(model)
```

So we have added the `MaxPool2d` layer to our CNN in PyTorch. Now that its added let us see the change in the output.

<p align="center"><img src='.\Images\Output_maxpooled2d.png' width=1000px></p>

Comparing the output of the `MaxPool2D` (above) model with the simple `ReLU` (below) model shows that there is now more emphasis on the edges detected. This is in line with the idea behind Max Pooling which is *to showcase the most prominent value for the window*.

<p align="center"><img src='.\Images\Relu_Ativated_Output.png' width=1000px></p>

Now that we now of the three basic convolution layers, we can proceed with learning about their implementation. Here is the [Convolution documentation](https://pytorch.org/docs/stable/nn.html#conv2d) from PyTorch for our guidance.

Also, its important to learn about the concept of *image augmentation*.

## Day 17: November 26, 2018

Approaching the 20 Day mark, I have a lot to do still but I have also covered a lot since. I will keep pushing.

We now move on to learning about the concept of styles and how it can be extracted from an image and eventually transferring these styles to another image. We can think of the *style* of an image as a certain unique feature that is recurring in an image. For example, certain brush strokes or certain patterns repeated throughout our image. This could be taken as styles. Since they are still features in the image, we can still train our network to look for them via filters. Knowing these filters would then allow us to apply a certain style from the *style image* on to a second image which is called the *content image*. The resulting output of this style transfer would be a third image which has the base image of the content image but with the applied filters to make it look a certain style.

What is the process of style transfer? This is the interesting part, we are using a CNN network (VGG19 for this case). We input our images, both input and output, to the VGG19 and we get their content representation. This *content representation* is simply the value of features taken deep within the CNN model. What this provides is the feature or think of it as the current style of the images. We get two content representations, one for the content image (input content) and another for style image (style content). The next step is we calculate the loss between our input content and our style content. By doing this we are actually getting the difference in style between between our target image and our style image. By knowing the loss we can then use optimizers to minimize the loss effectively transferring the style of our style content to the target image. In a way, VGG19 is not used as a classifier but as a feature extractor. Then once the feature is extracted a loss function is defined for the target image and the reference style and by using backpropagation we adjust the filters on the target image to match a certain representation of our style image.

Need to read more on Gram Matrix, they are the mathematical equivalent of the content representation of an image. Gram matrix are achieved by converting the 3D convolutional layers (2D image but with depth makes it 3D) to its 2D representation by sort of flattening it. Then this 2D matrix is multiplied by its transpose so that we get a square matrix. This matrix is then considered as our gram matrix. It contains all the features of the image up to that point. To get the the most style transfer of an image, we have to consider its large scale style up until the smallest style details it has. This will eventually lead us to having multiple gram matrices for representing the different dimensions of the image from the original and large scale style (generic) to the smaller scale style (minute). This allows us to copy the most style details.

Since gram matrices are value representations of the image and the style, we know that our loss function should be regression. They tend to use Mean Squared error for this application and we will be using that as well. This would allow us to adjust the weights of the target image to reflect a value closer to the style we want.

A final thing before we proceed is the ratio of our style to content. The idea behind this ratio is that we want to control *"how much"* style we want to transfer. The multiplier for content weight is denoted by $\alpha$. For the style weight, the multiplier is denoted by $\beta$. The ration between them is denoted as $\alpha / \beta$. The general idea is that $\beta$ values will always be larger than $\alpha$ which intuitively makes sense since we want to actually transfer the style. The trend is that the smaller the value of the ratio between the two weights, the more style we are transferring to our target image. One thing to not is that while we want to transfer as much style to our target as possible, we also have to consider how much style is too much. There will be a ratio where we can say that the style transfer has gone overborad but this is subjective to the effect that we want to achieve.

The objectives for today would be to do the practice exercises in CNN for PyTorch. These are already available in Colab as I have copied the repo into there.

## Day 18: November 27, 2018

Excited to do the CNN training exercises and transfer learning exercises. I just have to do finish up my APE today and I should be good to start these items.

In the meantime I have finished up the transfer learning videos. Up next is RNNs which is a bit short. Its just that and I think 3 more modules. I have not yet touched on the lab challenge but we will get there. I will be putting in the work. :frowning:

## Day 19: November 28, 2018

Slight change of plans. I am deep in work right now. There will be a slow down in progression in the coming days. :sob:

## Day 20: November 29, 2018

Day 20. What I accomplished yesterday was the Colab Code snippets repo. That is already done. For today, more on background reading for the competition. Plan is to read up Kernels from previous Kaggle Competitions.

For now I am reading this Kernel on Kaggle [Black & White CNN](https://www.kaggle.com/titericz/black-white-cnn-lb-0-782). I am finding some interesting boiler plates for use. I lifted some good blocks and pasted it below. It is a customizable input CNN pipeline. Simple loops in Python but it makes sense and makes life easier. :wink:

```python
def custom_single_cnn(size, conv_layers=(8, 16, 32, 64), dense_layers=(512, 256), conv_dropout=0.2,
                      dense_dropout=0.2):

    '''
    Source: https://www.kaggle.com/titericz/black-white-cnn-lb-0-782
    '''
    model = Sequential()
    model.add( Conv2D(conv_layers[0], kernel_size=(3, 3), padding='same', activation='relu', input_shape=(size, size, 1)) )
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    #if conv_dropout:
    #    model.add(Dropout(conv_dropout))

    for conv_layer_size in conv_layers[1:]:
        model.add(Conv2D(conv_layer_size, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        if conv_dropout:
            model.add(Dropout(conv_dropout))

    model.add(Flatten())
    if dense_dropout:
        model.add(Dropout(dense_dropout))

    for dense_layer_size in dense_layers:
        model.add(Dense(dense_layer_size, activation='relu'))
        model.add(Activation('relu'))
        if dense_dropout:
            model.add(Dropout(dense_dropout))

    model.add(Dense(NCATS, activation='softmax'))
    # NOTE: NCATS here is number of categories which is dependent on the dataset.
    return model

def top_3_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=3)

def top3_acc( tgt, pred ):
    sc = np.mean( (pred[:,0]==tgt) | (pred[:,1]==tgt) | (pred[:,2]==tgt) )
    return sc
```

This **WILL** make creating the model easier and it should still hold true for PyTorch. `custom_single_cnn` will create the the model for the CNN from the input up until the classification, __*neat*__.

### Sample use case

No need to define every single layer. :+1:

```python
STEPS = 500
size = 32
batchsize = 512

model = custom_single_cnn(size=size,
                          conv_layers=[128, 128],
                          dense_layers=[2048],
                          conv_dropout=False,
                          dense_dropout=0.10 )
model.compile(optimizer=Adam(lr=0.002), loss='categorical_crossentropy',
              metrics=[categorical_crossentropy, categorical_accuracy, top_3_accuracy])
print(model.summary())
```

Here is a quick excerpt from user `bestfitting` regarding the recently concluded Airbus Ship Detection Challenge.

>>Thank you! It's quite lucky to get top 3 in this competition. There are so many great and kind kagglers at kaggle,I am only a common one,there is no magic. According to my limited experiences on kaggle competitions,we all want to do more experiments during a limited time,there are a lot of ways:finding some teamates,getting more computation resources,and investing more time... In my humble opinion,the most important factor is efficiency. To improve efficiency,learning from others and learn from previous competitions are important. For example,my solution of this competition is quite similar to SeaLion competition,only replace the UNET model to SALT competition's.To save training time,I cropped 256x256 patchs where may containing ships from 768x768 images, just as I had done in sealion competition,so I can train models on a 1080Ti GPU easily And I try to do experiments on small and simple model and data as I did in CDiscount competition,it is very important in a competition with large dataset,I find res18 with 96x96 size input is enough for experiments in Draw competition,I am training model on my old machine with 4 Titan Maxwell GPUs :) Learning from others(absolutley including you @titericz) is very helpful to me,during SALT and this competition,I read all the solutions of related kaggle competitions,for example,DSB 2018(and I read your @dhammack solution of DSB2017 solution carefully after I entered Carvana Competition) so I can make sure I did not miss any skills and tricks related.By doing so,I can 'Teamup' with all the winners. As we know,the deep learning is very hot in both academic and industry,so I read papers everyday,ICCV,CVPR....,and search and read all related papers during a competition(>100 per competition often),what's more, I also read source codes of them or reproduce some of them(I find if we can be proficient in using Keras,pytorch,caffe,tensorflow,it will be more efficient,my experiences as an enthusiastic programmer helped me a lot,I read, modified a lot of software and built solutions on them,such as Android,Eclipse,Birt,Hadoop,Mahout,MySql...these names bring back so many bitter-sweet memories )When I need them in a competition,I can select some most promising one,for example,I find CBAM is a very good attention method,so I used it in SALT and this competition,and it's better than SE mechamism.By reading papers,we can 'Teamup' with most professional persons in the world in a certain field.I think these are also @hengck23 and @mihaskalic 's everyday pratices. As to my everyday life and job,I do physical exercises every day at least for an hour and I can arrange my work freely,so I can start a training and predicting batch and then launch a telephone meeting to discuss some projects in real life,I think they are much easier than kaggle and algorithms such as BERT--I'm reading now :)

I am learning a lot just by reading these discussion topics. The use of templates and previous challenges builds up our skills in competing. Some more additional readings from the user is [Fraud detection solution explanation](https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection/discussion/56262) and [Understanding Amazon from Space solution](https://www.kaggle.com/c/planet-understanding-the-amazon-from-space/discussion/36809).

There are a lot of good tutorials and knowledge competitions in Kaggle that are worth looking into [Titanic Tutorial](https://www.kaggle.com/mrisdal/exploring-survival-on-the-titanic). Then there is this Kernel on [Data Exploration in Python](https://www.kaggle.com/pmarcelino/comprehensive-data-exploration-with-python).

## Day 21: November 30, 2018

Today, the focus is in progressing on the RNN lesson. I am now on Character-wise RNN. I am still skipping on the notebooks but I will be back for them later. Right now the idea is to finish the lectures first, I already have watched some of them but I am excited to learn about implementation in PyTorch.

Also, in terms of progress in the Lab Challenge I have been reading up on Kaggle some discussions and Kernels. The lab challenge channel is also buzzing with activity. I have to reconnect with Rob and others to discuss this and possibly have some talk on their progression as well. For now, head down and watch on.

7:00 PM, I just finished the Character-wise RNN implementation module for PyTorch. The same concept holds true for this one, obviously. Its just the implementation in PyTorch that is different. We still have to consider the batches and sequence lengths and we are still going to get some gibberish results some times. All of those things can be improved, its the concept here and the implementation that counts.

So now I am on the Sentiment analysis using RNNs. This is an interesting topic because I am already looking at a use case for this after it is done. Something work related. :smirk:

I am also browsing the Final Project section. It's due in 1 month. Talk about pressure. The flow would be that the model has to be created, and then from there we have to save our checkpoints (deployment of PyTorch Models should cover this) and load that up to Udacity's notebook for evaluation. Basically, they want us to make use of the presented public data to build our model and then send them the checkpoints for evaluation. Aiming for **_TOP_** because what else is there to aim for.:dart:

[Resume writing tips](https://zety.com/blog/data-scientist-resume-example).

https://www.kaggle.com/mjbahmani/top-5-data-visualization-libraries-tutorial
https://www.kaggle.com/titericz
https://medium.com/neuralspace/kaggle-1-winning-approach-for-image-classification-challenge-9c1188157a86

## Day 22: December 1, 2018

I was just reading a post today which gave me a great new idea in my head. This is the [article](https://medium.com/neuralspace/kaggle-1-winning-approach-for-image-classification-challenge-9c1188157a86) in Medium. What I liked about it is that it gives a semblance of structure and with structure its easier to at least have direction.

### Study the Dataset

The outline is **first to study the data**, with the context of what the task is. We need to first analyze our data before we proceed with creating a model. We have to know how complex our data is, if it is balanced or unbalanced. This is where visualizing the dataset could come handy. What they recommend is to use t-Distributed Stochastic Neighbor Embedding to reduce dimensionality and visualize the data.

>>t-Distributed Stochastic Neighbor Embedding (t-SNE) is a technique for dimensionality reduction that is particularly well suited for the visualization of high-dimensional datasets. The technique can be implemented via Barnes-Hut approximations, allowing it to be applied on large real-world datasets.

Here is what t-SNE would look like for both the Kaggle Seedling dataset and MNIST dataset.

<p align="center"><img src='.\Images\TSNE-Kaggle-Seedlings.png' width=1000px></p>

<p align="center"><img src='.\Images\mnist_tsne.jpg' width=1000px></p>

What does t-SNE allow us to see? It shows us the difference between the classes of the data. It will allow us to know if it is difficult for humans or for machines to classify. If we look at the the seedling data set (first one), it shows that the classes are distributed and its quite difficult to classify them. Looking at the MNIST dataset (2nd image) shows that the numbers are fairly easy to classify. Another important thing the t-SNE might be able to provide is which classes tend to have similar traits and therefore much harder to distinguish.

One other basic method under the dataset analysis portion is looking at our class distribution. This would allow us to see if there is under-representation of classes in the dataset which will contribute to our accuracy as well. Below we see the initial distribution of our classes. The next image visualizes the distribution in our data and it clearly shows some under representation. We would be able to solve this on the next steps.

<p align="center"><img src='.\Images\kaggle-image-distribution.png' width=500px></p>

<p align="center"><img src='.\Images\kaggle-distribution-graph.png' width=500px></p>

Now that we are able to visualize the data we are now able to do some basic benchmarking test to it. The objective of this is that we want an accuracy and result that will be our basis of knowing whether we are moving in the right direction (lower loss, higher accuracy etc.). The first step in creating a benchmark would be to create a *training and validation set* which is covered in the documentation of PyTorch or SKLearn.

### Create a Benchmark

In step one we have covered what task is to be performed as well as analyze our data. In the sample of the article we are following, the objective is to classify seedlings from images. From here we know that this is a *classification problem* and that since we are dealing with image inputs we would want to use *Convolutional Neural Networks*.

A sample benchmark would be using the pre-trained network from the available networks in the library. Keras and torchvision has some pre-trained models in the library available for download and use. What our benchmark would be is to have a simple pre-trained model as feature extractor and build on top of it some fully connected layer. In the case of the author's benchmark, he used the Keras library pre-trained model in the ImageNet dataset and he slowly fine tuned it to his task. Note, we are using ImageNet trained models here because it is the best fit to our task. ImageNet would have covered classifying plants or flowers. Using MNIST in here would have been a bad choices since those are handwritten digits which has nothing to do with seedlings, or plants. **[TL:DR]** _Knowing the background of the models would be to your advantage so try reading their papers as well_.

Back to the author's implementation, he used two pre-trained models: ResNet50 and InceptionResNetV2 for his task. The reason for choosing two models is that it allowed him to benchmark the dataset on one simple and one high end model to understand if it is overfitting/underfitting the dataset on the model.

<p align="center"><img src='.\Images\Benchmark-Kerasmodels.png' width=800px></p>

From the authors implementation, he used a pre-trained model and removed the last output layer. From that he added an output of 12 nodes to provide the probability distribution output of the classes.

> The goal of the benchmarking step is for us to know if we can improve our model by testing it out on some pre-trained models and slowly un-freezing layers for additional fine-tuning. The end-game would therefore be a tabulated result of some runs on different variations of the model and their accuracy and other additional metrics.

### Data cleaning and augmentation

Once we have an idea of what the benchmark scores are we can then move on to improving the dataset. One way to do this is via *data augmentation*. Also, we can take on fixing the dataset. We know that the dataset is not balanced so we can do some balancing first.

> Note: Real life datasets would rarely be balanced and this would mean that the performance of a model over a minority class is not that good. The cost of misclassifying a minority class example to a normal example is much higher than the cost of a normal class error. (from the original article)

The article provides us two ways on how to deal with unbalanced data.

1. **Adaptive synthetic sampling approach for imbalanced learning (ADASYN)** - This would generate additional synthetic data for classes with less samples. This augmentation is done based adaptively based on how difficult the class is to learn compared to other samples.

>The essential idea of ADASYN is to use a weighted distribution for different minority class examples according to their level of difficulty in learning, where more synthetic data is generated for minority class examples that are harder to learn compared to those minority examples that are easier to learn. As a result, the ADASYN approach improves learning with respect to the data distributions in two ways: (1) reducing the bias introduced by the class imbalance, and (2) adaptively shifting the classification decision boundary toward the difficult examples. [5](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=4633969&tag=1)

2. **Synthetic Minority Over-sampling Technique (SMOTE)** - This involves oversampling of minority classes and undersampling majority classes to get best results.

>A combination of our method of over-sampling the minority (abnormal) class and under-sampling the majority (normal) class can achieve better classifier performance (in ROC space) than only under-sampling the majority class.[6]()

<p align="center"><img src='.\Images\Kaggle-Smote-Adasyn.png' width=800px></p>

## Day 23: December 2, 2018

**Continuation of the pipeline from yesterday:**

Once we are able to get the better algorithm to balance the dataset, we are going to augment our data. Remember that balancing first before augmentation. In terms of augmentation we have various options which can be located at the [documentation for transforms](https://pytorch.org/docs/stable/torchvision/transforms.html). We can do scaling, cropping, rotation, flipping, off-centered crop, light-condition changes  and translation as well as using GANS. Once we are able to augment our data, we can proceed with the next step.

### Hyperparameter Optimization

Since we have a benchmark, a balanced and augmented data, we can proceed with playing with our hyperparameters. We can play with our learning rates and adjust them. One idea presented in the article was the use for **cyclical learning rates**. The idea behind it is that the learning rate gets change within a range of values instead of consistently decreasing. *Note that this is also a good time to record as much as we can in terms of the results so that we can graph them*. An example graph below shows that the loss starts to increase again past the $10^{-1}$ learning rate.

<p align="center"><img src='.\Images\LR-VS-Loss.png' width=500px></p>

Once we are able to compute for the losses of the model, we can actually create new models and merge them together. This technique is called **Ensembling** and it is very popular in competition as it produces great results instead of a single model. The author went on to explain how he implemented **snapshot ensembling** [paper](https://arxiv.org/abs/1704.00109).

Once the author found the learning rates, he set it down and started playing with the image size. Note that the goal of the learning rate experimentation is to find which learning rates converge best and give the better result. So back to playing around with image size, we can actually play around with the size of the image we input. If you think about it a larger input size will mean more pixels to play with and therefore more data and feature to possibly extract. The other face of the coin is that it is computationally expensive. For this reason, we have to also keep track of whether our changes in image size can lead to better result or is it just a wasted increase with little bearing to the metrics of the model. For the author's implementation he chose to train it on a 64x64 image size over ImageNet, unfreeze some layers, apply cyclic learning rate and snapshot ensembling, take the weights and change the size to 299*299 fine tuned the weights from the 64x64 model and do snapshot ensembling and learning rate with warm restarts.

His implementation is well advanced but it should be doable in our case.

### Result Visualization

The final step to take would be to visualize the results. This will allow us to check with class are best and worst performers and take actions to improve their results if possible.

A good way to understand the result is via the confusion matrix.

>In the field of machine learning and specifically the problem of statistical classification, a confusion matrix, also known as an error matrix, is a specific table layout that allows visualization of the performance of an algorithm, typically a supervised learning one (in unsupervised learning it is usually called a matching matrix). Each row of the matrix represents the instances in a predicted class while each column represents the instances in an actual class (or vice versa). The name stems from the fact that it makes it easy to see if the system is confusing two classes (i.e. commonly mislabeling one as another).

<p align="center"><img src='.\Images\kaggle-confusion-matrix.png' width=800px></p>

As we can see above we have the confusion matrix of the predicted and true class for our model. Ideally, it should all bee high in the diagonal which would mean that the predicted class is the same as the true class. Even if the confusion matrix is not correct we can still use it as a basis in improving our model. We can find out which class is performing bad and try to augment that and retrain our model for that.

Once we are happy with all our model's results, we merge the validation and training dataset into one. We make one last pass for our network in the merged set and we use that final model to test out the test dataset and see our results.

### T-SNE

Let us briefly read about [t_SNE](https://lvdmaaten.github.io/tsne/). It is a technique for dimensionality reduction that is well suited for visualization of high-dimensional datasets. In the author's [blog](https://lvdmaaten.github.io/tsne/) there are some implementations of the t-SNE on several languages and some FAQs as well. I suggest reading the original paper for more guidance.

I found this [tutorial on GitHub](https://github.com/oreillymedia/t-SNE-tutorial).

<p align="center"><img src='https://raw.githubusercontent.com/oreillymedia/t-SNE-tutorial/master/images/animation.gif' width=800px></p>

## Day 24: December 3, 2018

I was just watching a YouTube video regarding t-SNE and its explanation. The example I can give for t-SNE is that it treats it like electrons (but of different charges not just + and -). What we mean here is that opposite points repel and similar points attract just like electrons. What happens is that at first we have a pile of points, as time goes on (i.e. model training) the points are able to know what class they are so they get attracted to those while at the same time getting repelled by other classes. This will go on for some time before the points are able to co-locate together with similar points and repel others of different classes.

### Taking on the challenge

:heavy_check_mark: First up would be downloading the data to Drive. I need to do this once to have my data on my Google Drive. I have also unzipped the files so that I can work with it.

```python
# Downloading to Google Drive
%cd 'My Drive/Colab Notebooks/pytorch_challenge'
!pwd
!ls
!wget https://s3.amazonaws.com/content.udacity-data.com/courses/nd188/flower_data.zip
!unzip flower_data.zip
```

## Day 25: December 4, 2018

There are some mixup on the date. It is day 25 for December 4. A few days more to go. So we really need to have started progress in the Challenge portion. Right now I have been able to download the dataset to the Drive. Next step would be to load the images.

```python
'''
    Source: https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
    Load data section
'''
# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

print("Initializing Datasets and Dataloaders...")

# Create training and validation datasets
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
# Create training and validation dataloaders
dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in ['train', 'val']}

# Detect if we have a GPU available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
```

### General Checklist

* [x] - Finish up Transfer learning videos :dart:
* [ ] - CNN training exercises :dart:
* [x] - Create a repository for Colab code snippets :gem:
* [ ] - Write up one pager, contact careers for inputs :date:

### Pipeline ideas and reading materials

:gem: Kaggle Competition [get started tutorial](http://blog.kaggle.com/2018/08/22/machine-learning-kaggle-competition-part-one-getting-started/)

:dart: Free [Machine Learning](https://www.kaggle.com/learn/machine-learning) course and [SQL](https://www.kaggle.com/learn/sql)

:bomb: [Deep Learning with Pytorch](https://medium.com/@josh_2774/deep-learning-with-pytorch-9574e74d17ad), a medium post from Josh Bernhard detailing the flow of the final Project.

:bulb: Anomaly detector in sequences (First is text, then images)

:clipboard: How do we publish our weights into production? Like the auto-assign, or anomaly detector. Worth figuring out. [LINK](https://medium.freecodecamp.org/a-beginners-guide-to-training-and-deploying-machine-learning-models-using-python-48a313502e5a), for deploying the model via Flask.

:hocho: [Deploying ML Models](https://towardsdatascience.com/deploying-deep-learning-models-part-1-an-overview-77b4d01dd6f7). From what I initially read, its possible via Kubernetes.

:bulb: SQL/noSQL for Database management and data wrangling.

:hocho: There are a lot of examples/tutorials in PyTorch website  for projects.

:bomb: [MRI brain images reconstruction](https://www.datacamp.com/community/tutorials/reconstructing-brain-images-deep-learning)