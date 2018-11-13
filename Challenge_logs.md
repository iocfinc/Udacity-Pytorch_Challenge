# Pytorch Challenge Logs

## Day 1: November 10, 2018

Official start of the challenge. Mostly doing some readings on what the challenge format would be. There would be an intro to pytorch. Then there would be the usual, CNNs and RNNs for DL with PyTorch as the framework. Finally, for the final project (which would be the basis of the course) we would be using transfer learning to come up with an image detection and classifier for flowers (orchids by the looks of it).

## Day 2: November 11, 2018

So today, I have started on the initialization for the challenge. For one, I have joined the [Slack Channel](https://pytorchfbchallenge.slack.com/messages/CDB3N8Q7J/convo/CDB3N8Q7J-1541904940.926900/), its already active and there are already some questions posted. For now I think I can help since I have some experience in the Deep Learning Nanodegree. Also, I have installed pytorch and torchvision as pre-requisites to the course.

Right now I am watching the interview with Soumith Chintala, one of the creators of PyTorch, regarding the history and uniqueness of PyTorch from other frameworks. For one thing, its approach was Python first meaning that the python ways we already now and want are applied to the system. It also has a JIT compiler which bridges the known Deep learning frameworks, caffe, tensorflow, torch, etc.,to be able to convert from one framework to another and also to a deployment ready C code for production. In terms of additional features, the PyTorch team is looking into support for Google Collab (for the free GPUs), more interactive notebooks for trainings and examples and also the use of tensorboard for PyTorch.

Right now its 2:23 PM, I have to pause for this session. I have watched the introduction as well as the introduction to pytorch videos so those are done. I have to do other things for next week but I'll be back probably later tonight. Objective for today is to consume the next lesson which is the Introduction to Pytorch (coding) by Mat Leonard (?).

So now, its 9:00PM. Back at it again. For now the idea is to setup Collab for the notebooks. I found [Collab](https://colab.research.google.com/notebooks/welcome.ipynb#recent=true) and it looks like there is an option for the use of a repository in github. Seems easy. So for now, I'll move over through the Introduction lessons and see what the first lab would be.

So what to expect:

* [x] - Tensors - The data structure of PyTorch
* [ ] - Autograd which is for calculating Gradients in NN training.
* [ ] - Training of an NN using PyTorch.
* [ ] - Use of PyTorch for Transfer Learning for image detection.

So first up was tensors in PyTorch. I thought tensors was some sort of proprietary naming of PyTorch, it was not. Its basically referring to the unit of tensor. So after that we went on to discuss `torch.mm` which is the matrix multiplication equivalent of `np.matmul` in torch. Also there is `torch.sum` which can also be called as a method `.sum()` which obviously sums up the values inside it. One important piece of information that was given in the introduction was the use of memory between numpy and torch. Obviously, pytorch will have compatibility with numpy so anything (an array for example) defined in numpy can be ported to torch via `torch.from_numpy` and vice versa via `.numpy`. In these operations, the memory used for the array are one and the same. Meaning that an operation done in an array that was ported to torch will also be reflected in the version of numpy since they are at the same memory. Also, the transpose operation `.T` is not used in torch. Instead, to match the dimensions of matrix multiplication, we are advised to use `.reshape(a,b)`, `.resize(a,b)` or `.view(a,b)` operation. It is highly advised to make use of `.view(a,b)` than the other two as they do have some issues according to Mat Leonard.

So here is an interesting trick for Collab and Google Drive mount. This should help in uploading those modules or python scripts like unit tests and others to your notebook. It will allow you to read from your google drive input files and others as well. A useful tip, you can run bash commands directly in the notebook via `!` command so `!ls` should output the list of files in your current drive. Which is neat.

```python
# TODO: The code below would start the initialization of your mounting of Google Drive to a Collab notebook.
!apt-get install -y -qq software-properties-common python-software-properties module-init-tools
!add-apt-repository -y ppa:alessandro-strada/ppa 2>&1 > /dev/null
!apt-get update -qq 2>&1 > /dev/null
!apt-get -y install -qq google-drive-ocamlfuse fuse
from google.colab import auth
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

In the code block above, we are creating a new class using the torch. We are calling the nn module and from that our `Network` will inherit its `nn.Module`. It is **mandatory** to inherit the `nn.Module` for our class. This will allow us to create the neural network on torch. For the initialized values we need to define the layers and the transformation we want which in this case is linear, I am thinking we can call Convolution or Recurrent later. Then there are the activation functions `nn.Sigmoid()` and `nn.Softmax(dim=1)`. It is important to take note of one unique property in PyTorch (or torch?) which is broadcasting. The argument `dim=1` for the softmax function is used to indicate which way the softmax is applied which in this case is on the column, setting it to 0 would mean that it is on a row. This is also important on other operations as well like division `A/B` which needs to have a `dim` argument with it. Finally, we can compile our model simply by calling out the `Network` class and calling model will simply print out the summary of the model we have created, more like `model.summary()` in Keras.

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

Now on exercise number 3 of 8: Training Nerual Networks. First off [here](https://pytorch.org/docs/stable/index.html) is the link for the documentation of PyTorch. It comes in handy in getting a deeper understanding of what is being discussed in the notebooks. It has more explanation on the process of [autograd](https://pytorch.org/docs/stable/notes/autograd.html#how-autograd-encodes-the-history). I was reading about the autograd feature and its quite intuitive actually and it is what allows PyTorch to be faster, especially in distributed computation (GPU/CUDA). Its whats going to allow computation of gradients during every pass, which in turn would allow us to manually set a layer to not update (for example, transfer learning).

After autograd we move into [Broadcasting semantics](https://pytorch.org/docs/stable/notes/broadcasting.html) for PyTorch. Apparently, its an inheritance of the NumPy broadcasting semantics. It is simply solving or pointing out dimensionality match or mismatches when we operate tensors (which we will as we progress). One possible way to reduce this error/complication is to understand the basics of Matrix Multiplication, mostly that it is not commutative (orders matter especially in determining the shape of the output).

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

Now that there is a model that was trained, our weights should now be updated an our model should now be good to go. Below is a sample of a way we can use the trained model for checking of results. As we can see we need to turn off the gradient updates. This is done because we are do not intend to train our network, we just want to try to pass a sample image to it. Without calling on the `no.grad()` function, the model will acutally consume memory even if we are not doing any operations.

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

! wget https://github.com/iocfinc/deep-learning-v2-pytorch/blob/master/intro-to-pytorch/helper.py  #Insert the  URL for the file
```

Use the code below to install PyTorch, alternatively just use code snippets.
```python
#NOTE: Use the code below to install PyTorch in Collab
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

Currently having some issues downloading the dataset from Fashion MNIST and MNIST via collab or local. Something about an OSError not reading the correct files from the link. Posted it on the Slack channel and got some responses on how to resolve it. They said to try downloading the file again as it might be corrupted which could explain the issue. As a workaround, I opened up the Python Terminal and ran the code from there. Interesting enough, it was able to download the files for the data set. I am not sure why it was throwing a non-gzip file error when I ran it on the notebook. But at least that is resolved. For now, more exercises on training the neural network for Fashion Mnist data set classification.

With regards to using PyTorch in collab, [here](https://cloud.google.com/blog/products/ai-machine-learning/introducing-pytorch-across-google-cloud) is a link detailing how this could be achieved (**in the future**) for now, TPUs are only for TensorFlow in collab which does make sense seeing as they are both Google managed. The good thing is that there is active collaboration between engineers to allow PyTorch the use of TPUs on Collab. Also, this is a [tutorial](https://github.com/nataliasverchkova/on-using-google-colab) for using Google Colab. Here are some more resources. This one is [about using [the GPU in Colab](https://medium.com/deep-learning-turkey/google-colab-free-gpu-tutorial-e113627b9f5d). Then we have [this one](https://jovianlin.io/pytorch-with-gpu-in-google-colab/) which is basically a starters guide on Colab and how to use it (brief explanation).

Now let us move on to inference and validation.

* [x] - Tensors - The data structure of PyTorch
* [x] - Autograd which is for calculating Gradients in NN training.
* [x] - Training of an NN using PyTorch.
* [ ] - Use of PyTorch for Transfer Learning for image detection.
* [ ] - Figure out using collab for the challenge. There is a GPU and TPU service on the cloud. :joy:
