# Lab Challenge Notebook

This notebook is for logging in the notes I have for the main lab Challenge of the scholarship. Its going to be only limited to materials used and methods done on the lab challenge. The progressing of the whole scholarship challenge would still be recorded on the [Challenge_logs.md](https://github.com/iocfinc/Udacity-Pytorch_Challenge/blob/master/Challenge_logs.md).

## December 4, 2018 - Day 25 of the Challenge

Just to recap:

:heavy_check_mark: Clone the repository to Google Drive
:heavy_check_mark: Download the dataset to Google Drive

The objective now is to create the `dataset` and `dataloader` functions for PyTorch to interface with the images.

As I said before, the community in Slack is great. I found this [guide from other users](https://docs.google.com/document/d/1-MCDPOejsn2hq9EoBzMpzGv9jEdtMWoIwjkAa1cVbSM/edit#heading=h.nj23sjpj5u97) about [FAQs and common issues](https://github.com/ishgirwan/faqs_pytorch_scholarship/blob/master/Lab.md) in the course.

Another helpful guide with regards to loading the dataset is [this notebook](https://colab.research.google.com/drive/1iDwVOoVBkuljUadRSCChs33iNtbrI6Wv#scrollTo=qn8t--2ttqDz).

So there was some brain-fart moment :smirk: during the loading of the dataset. Then I noticed that I had [this resource](https://medium.com/@josh_2774/deep-learning-with-pytorch-9574e74d17ad) flagged beforehand. Very interesting read. It is very informative and for the most part, covers most of the problems in the challenge.

:gem: [Josh's Guide on PyTorch Project.](https://medium.com/@josh_2774/deep-learning-with-pytorch-9574e74d17ad)

### Simple Data Exploration

The code below would be listing the contents of the folders in the Working Directory. It still has to be re-parsed again for Colab use.

```python
# NOTE: Checking for the number of items in the data:
# Augment this for colab
!for dir in ./*/
do 
ls ./${dir} -l . | egrep -c '^-'
done
```

### Transformation and Augmentation

In this section we are going to define the transformation we will be doing to the data. Most important would be the correct normalization parameters since we are using PyTorch pre-trained models, also the input size needs to be flipped. Additional data and transforms can be done later.

Also, note that this is going to be an imbalanced dataset so we need to fix it via ADASYN possibly later.

```python
normalize = transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]) # Specific for PyTorch Pre-trained

data_transforms = {
    'train' : transforms.Compose([
        transforms.RandomRotation(90), # Randomized rotation until 90 degrees, to avoid fills.
        transforms.RandomResizedCrop(224), # Required to be 224 size for pre-trained
        transforms.RandomHorizontalFlip(), # Random flipping
        transforms.ToTensor()
        normalize
    ]),
    'valid' : transforms.Compose([
        transforms.Resize(256), # Resized image to 256
        transforms.CenterCrop(224), # Required to be 224 size for pre-trained, Centered
        transforms.ToTensor()
        normalize
    ]),
    'test' : transforms.Compose([
        transforms.Resize(256), # Resized image to 256
        transforms.CenterCrop(224), # Required to be 224 size for pre-trained, Centered
        transforms.ToTensor()
        normalize
    ])
    # NOTE: valid and test need to be the same, the idea is that validation is mini-tests per epoch so it makes sense.

}
```

### Dataloader

In here we are going to load the image dataset for PyTorch use. We will also be defining the batch sizes and check the class names for our Data.

```python
# NOTE: Need to edit this to conform with the folder, structure
train_dir = # NOTE: Add the original locations from the provided notebook.
valid_dir = # NOTE: Add the original locations from the provided notebook.
dirs = {
    'train' : train_dir,
    'valid' : test_dir
} # We are using this for simpler loading
# Set the datasets
image_datasets = {x: datasets.ImageFolder(dirs[x],transform=data_transforms[x]) for x in ['train','valid']}
# dataloaders
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size = 32, shuffle = True) for x in ['train','valid']}

class_names = image_datasets['train'].classes

# NOTE: The expected class names are not yet mapped.
```

### Transfer Learning

Start of actual PyTorch model training. I am thinking of using ResNet50 and InceptionV3 for baselines.

```python
model = models.vgg19(pretrained = True) # This is a lift from Josh's post. Just for sanity checking first.

model # So that we can see the architecture, it should be simple enough since there are only 19 layers.
```

### Defining the Classifier (Feature Extractor Method)

Here we are simply adding on top of the pre-trained model. For better results I intend to use fine-tuning later on.

```python
# TODO: Define our classifier that will be replacing the last layer in VGG19.
classifier = nn.Sequential(
    OrderedDict([
        ('fc1',nn.linear(25088,4096)),
        ('relu',nn.ReLU()),
        ('fc2',nn.Linear(4096,102)),
        ('output',nn.LogSoftmax(dim = 1))
    ])
)
```

### Locking the Parameters (Feature Extractor Method)

In here we need to lock all previous model parameters that have auto-grad before we call on our created classifier.

```python
for param in model.parameters():
    param.requires_grad = False
```

### Replacing the Classifier layer

Since we are using VGG19 as a feature extractor, we can proceed now with replacing the final layer to the correct output dimension of our use case.

```python
# NOTE: It just happened that the last layer of VGG19 is called classifier. Be careful in calling this on other models.

model.classifier = classifier
```

What is happening is that the layer in the model (VGG19) called classifier is being replaced by our classifier. It may be  the case that other models loaded will have a different final layer name.

### Mapping of Names

```python
{"21": "fire lily", "3": "canterbury bells", "45": "bolero deep blue", "1": "pink primrose", "34": "mexican aster", "27": "prince of wales feathers", "7": "moon orchid", "16": "globe-flower", "25": "grape hyacinth", "26": "corn poppy", "79": "toad lily", "39": "siam tulip", "24": "red ginger", "67": "spring crocus", "35": "alpine sea holly", "32": "garden phlox", "10": "globe thistle", "6": "tiger lily", "93": "ball moss", "33": "love in the mist", "9": "monkshood", "102": "blackberry lily", "14": "spear thistle", "19": "balloon flower", "100": "blanket flower", "13": "king protea", "49": "oxeye daisy", "15": "yellow iris", "61": "cautleya spicata", "31": "carnation", "64": "silverbush", "68": "bearded iris", "63": "black-eyed susan", "69": "windflower", "62": "japanese anemone", "20": "giant white arum lily", "38": "great masterwort", "4": "sweet pea", "86": "tree mallow", "101": "trumpet creeper", "42": "daffodil", "22": "pincushion flower", "2": "hard-leaved pocket orchid", "54": "sunflower", "66": "osteospermum", "70": "tree poppy", "85": "desert-rose", "99": "bromelia", "87": "magnolia", "5": "english marigold", "92": "bee balm", "28": "stemless gentian", "97": "mallow", "57": "gaura", "40": "lenten rose", "47": "marigold", "59": "orange dahlia", "48": "buttercup", "55": "pelargonium", "36": "ruby-lipped cattleya", "91": "hippeastrum", "29": "artichoke", "71": "gazania", "90": "canna lily", "18": "peruvian lily", "98": "mexican petunia", "8": "bird of paradise", "30": "sweet william", "17": "purple coneflower", "52": "wild pansy", "84": "columbine", "12": "colt's foot", "11": "snapdragon", "96": "camellia", "23": "fritillary", "50": "common dandelion", "44": "poinsettia", "53": "primula", "72": "azalea", "65": "californian poppy", "80": "anthurium", "76": "morning glory", "37": "cape flower", "56": "bishop of llandaff", "60": "pink-yellow dahlia", "82": "clematis", "58": "geranium", "75": "thorn apple", "41": "barbeton daisy", "95": "bougainvillea", "43": "sword lily", "83": "hibiscus", "78": "lotus lotus", "88": "cyclamen", "94": "foxglove", "81": "frangipani", "74": "rose", "89": "watercress", "73": "water lily", "46": "wallflower", "77": "passion flower", "51": "petunia"}
```

## December 5, 2018 - Day 26 of the Challenge

For further dataset exploration we can [use this paper](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/). This is actually part of the Visual Geometry Group's studies and its way back in 2008. They are obviously the ones behind the VGG architecture. From the [page] of the 102 Category Flower Dataset, we are able to see that the dataset is indeed imbalanced.

```python
# Helper function for training
def train_model(model, dataloaders, criterion,optimizer,num_epochs = 20):
    start = time.time()
    val_acc_history = [] # For plotting later
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0 # Initialized at 0
    for epoch in range(num_epochs):
        epoch_start = time.time()
        print('Epoch {}/{}'.format(epoch, num_epochs-1))
        print('-' * 10)

        for phase in ['train','val']: # Distinguish first if training or validation to turn off the updates
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase =='train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs,1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                    # Do updates of loss
                        print('Updating loss')
                        loss.backward()
                        optimizer.step()
                print('Updating running loss')
                running_loss +=loss.item()*inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            print('Updating epoch loss')
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double()/ dataset_sizes[phase]
            epoch_elapsed = time.time() - epoch_start
            print('Epoch completed in {:0f}m {:0f}s'.format(epoch_elapsed //60, epoch_elapsed % 60))

            print('{} Loss: {:.4f}  Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            # NOTE: For updating the best model
            if phase =='val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase =='val':
                val_acc_history.append(epoch_acc)
        print()
    time_elapsed = time.time() - start
    print('Training completed in {:0f}m {:0f}s'.format(time_elapsed //60, time_elapsed % 60))
    print('Best val Acc: {:.4f}'.format(best_acc))
    # Return the model with the best weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history

```

We now have the code block for training. The problem right now is that the network is training for a while. I am not sure this should be the case as the runtime should have been GPU enabled.

I need to pause. Its stuck for some reason and I feel like I'm banging my head against the wall. :cry:. I'll pause for a bit and resume this one later.

## December 6, 2018 - Day 27 of the Challenge

New plan of attack, divide and compile piece by piece instead of one entire helper function.

And no progress was done today. Same thing will happen until next week.

## December 10, 2018 - Day 31 of the Challenge

```python
start = time.time()
val_acc_history = []
best_model_wts = copy.deepcopy(model.state_dict())
best_acc = 0.0 # Initialized at 0
for epochs in num_epochs:
    epoch_start = time.time()
    print('Epoch {}/{}'.format(epoch, num_epochs-1))
    print('-*-' * 10)
    for phase in ['train','valid']: # Distinguish first if training or validation to turn off theupdates
        if phase == 'train':
            model.train()
        else:
            model.eval()
        running_loss = 0.0
        running_corrects = 0
        for inputs, labels in dataloaders['train']:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            _, preds = torch.max(outputs,1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.steps()
            running_loss +=loss.item()*inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
        print('Updating epoch loss')
        epoch_loss = running_loss / dataset_sizes[phase]
        epoch_acc = running_corrects.double()/ dataset_sizes[phase]
        epoch_elapsed = time.time() - epoch_start
        print('Epoch completed in {:0f}m {:0f}s'.format(epoch_elapsed //60, epoch_elapsed %60))
        print('{} Loss: {:.4f}  Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
        # NOTE: For updating the best model
        if phase =='valid' and epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = copy.deepcopy(model.state_dict())
        if phase =='valid':
            val_acc_history.append(epoch_acc)
        print()
    time_elapsed = time.time() - start
    print('Training completed in {:0f}m {:0f}s'.format(time_elapsed //60, time_elapsed % 60))
    print('Best val Acc: {:.4f}'.format(best_acc))
    # Return the model with the best weights
    model.load_state_dict(best_model_wts)

```

I am giving this set above one last setup before I scratch it and start something new. It is not even training properly by the looks of it. I will leave it at 1 hour and check if it will even go to the second epoch. As it stands its already eating up 15 minutes.

## December 11, 2018 - Day 32 of the Challenge

Left the notebook to run training and slept in. I woke up and there is no result. Somehow there is an issue with the looping I believe, it just sits there stuck at epoch 1. It got disconnected from the session.

Since it is obviously not working, I need a rethink of what could be wrong.

I think I may have figured it out. Its quite minute actually. I am thinking it had something to do with how I called my directory. I merged strings instead of `os.path`. Right now it is training. Seriously though. Was that it?

Since it is now running, (FINALLY), I can shift my focus to creating the baseline and measurements. First up would be timings for speed measurements. Second would be recording the accuracy. Third would be to actually implement a very complicated model as well (InceptionV3) to check for over and under-fitting.

[Verification OpenSource](https://github.com/GabrielePicco/deep-learning-flower-identifier)

So first result:
```python
# NOTE: Current Parameters
# Batch size
# Criteria NLLLoss which is recommended with Softmax final layer
criteria = nn.NLLLoss()
# Observe that all parameters are being optimized
optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)
# Decay LR by a factor of 0.1 every 4 epochs
sched = lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.1)
# Number of epochs
eps=5
```
For VGG19 - Pre-trained Fixed feature extraction model.
```python

Epoch 0/4
----------
train Loss: 2.6440 Acc: 0.4539
valid Loss: 0.9257 Acc: 0.7421

Epoch 1/4
----------
train Loss: 1.1459 Acc: 0.6821
valid Loss: 0.7499 Acc: 0.7897

Epoch 2/4
----------
train Loss: 0.9540 Acc: 0.7376
valid Loss: 0.7065 Acc: 0.8068

Epoch 3/4
----------
train Loss: 0.8814 Acc: 0.7582
valid Loss: 0.6084 Acc: 0.8435

Epoch 4/4
----------
train Loss: 0.6235 Acc: 0.8237
valid Loss: 0.3788 Acc: 0.8949

Training complete in 19m 12s
Best val Acc: 0.894866
```

Now that it works, its all now up to improving it. I am interested in doing this mobile. Possibly create it using the mobilenet. It should train fast. Its interesting.

## December 12, 2018 - Day 33 of the Challenge

Right now I am moving on to baseline for Resnet152. Same idea as before, just changed some lines a bit to be sure that it matches the Resnet model.

I actually have to abandon Resnet152 for now, it takes a long time. I am using Resnet18 as the first baseline. Re-computing again the time it will take. Below we have the results of the baseline for Resnet18. I had to rethink my overall strategy. By the looks of it, the first epochs take most of the time. It still converges though so that is good. At least now I know that I just need to have more patience in dealing with the first epoch. Also, The difference in size between VGG19 and Resnet18 is night and day. Their results are almost the same but the savings in space is what's selling the idea of Resnet18 to me. I want to check out Resnet50 then Resnet152 later. It is doable.

```python
Epoch 0/4
----------
Epoch completed in 12.000000m 12.717537s
train Loss: 3.0937 Acc: 0.3381
Epoch completed in 15.000000m 33.946764s
valid Loss: 1.3286 Acc: 0.6993

Epoch 1/4
----------
Epoch completed in 1.000000m 42.412760s
train Loss: 1.3900 Acc: 0.6560
Epoch completed in 1.000000m 56.414605s
valid Loss: 0.6988 Acc: 0.8325

Epoch 2/4
----------
Epoch completed in 1.000000m 42.071889s
train Loss: 0.9681 Acc: 0.7497
Epoch completed in 1.000000m 55.975896s
valid Loss: 0.5092 Acc: 0.8729

Epoch 3/4
----------
Epoch completed in 1.000000m 41.956042s
train Loss: 0.8095 Acc: 0.7824
Epoch completed in 1.000000m 55.861529s
valid Loss: 0.4542 Acc: 0.8753

Epoch 4/4
----------
Epoch completed in 1.000000m 42.830410s
train Loss: 0.6662 Acc: 0.8304
Epoch completed in 1.000000m 56.911821s
valid Loss: 0.3498 Acc: 0.9156

Training complete in 23m 19s
Best val Acc: 0.915648
```

The convergence of the model for different rates would be the next problem that we should solve. This could be helped by the adjustment of the LR in a way, also we would want to add the graph of the losses for the epochs and the LR to make sure that they are still decreasing. This is a simple list for the code change. Very much doable. For now I have to go home, its 9:00 AM. I have more to do and I need sleep. Just happy that I am chipping away some progress on this one.


```python
# NOTE: From style transfer lecture:

'''
I am copying the content of the style transfer features. I am planning on reusing the method here in training the conv layers only.
'''

def get_features(image, model, layers=None):
    """ Run an image forward through a model and get the features for 
        a set of layers. Default layers are for VGGNet matching Gatys et al (2016)
    """
    
    ## TODO: Complete mapping layer names of PyTorch's VGGNet to names from the paper
    ## Need the layers for the content and style representations of an image
    if layers is None:
        layers = {'0': 'conv1_1',
                  '5': 'conv2_1', 
                  '10': 'conv3_1', 
                  '19': 'conv4_1',
                  '21': 'conv4_2',  ## content representation
                  '28': 'conv5_1'}
        
    features = {}
    x = image
    # model._modules is a dictionary holding each module in the model
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x
            
    return features
```

```python
'''
Example of how to freeze the layers via names.
'''
net.fc2.weight.requires_grad = False
net.fc2.bias.requires_grad = False
```

## December 15, 2018 - Day 36 of the Challenge

This is becoming uncharacteristically long missing daily updates. This has been a very hectic week. With Charlie acting up and monopolizing my time. Missing my updates on other tickets. Missing progress in the project. Missing home. This is really tiring. But a decision has been made. Next step is to carry it out and arrange for the contingencies. Right now I will be going home for the next few weeks. There will be time to do more of the challenge. The plan has been written on paper and shall be carried out.

What is next for the project? **Finish up the entire pipeline first**. I was thinking about it wrong. The idea is to also get the results not just the accuracy. The image plotting and confusion matrix are in the later part of the notebook so we need to get there first. There is a sample of the notebook [here](https://github.com/GabrielePicco/deep-learning-flower-identifier). This should help.

Now we are training RESNET18 at 10 epochs. This is just to check if there would be an improvement in the accuracy or if it will saturate.

```python
'''
Here are the results.
Setup was: LR = 0.001, Resnet18, 10 epochs, Batchsize = 32
'''

cuda
Epoch 0/9
----------
Epoch completed in 7.000000m 45.221650s
train Loss: 1.6388 Acc: 0.5707
Epoch completed in 12.000000m 43.761453s
valid Loss: 1.1383 Acc: 0.6993

Epoch 1/9
----------
Epoch completed in 1.000000m 44.877961s
train Loss: 1.5013 Acc: 0.6009
Epoch completed in 1.000000m 59.837174s
valid Loss: 1.0305 Acc: 0.7262

Epoch 2/9
----------
Epoch completed in 1.000000m 49.457006s
train Loss: 1.3347 Acc: 0.6462
Epoch completed in 2.000000m 4.468249s
valid Loss: 0.9435 Acc: 0.7298

Epoch 3/9
----------
Epoch completed in 1.000000m 48.817481s
train Loss: 1.2809 Acc: 0.6587
Epoch completed in 2.000000m 3.711036s
valid Loss: 0.8501 Acc: 0.7567

Epoch 4/9
----------
Epoch completed in 1.000000m 47.415198s
train Loss: 1.1599 Acc: 0.6954
Epoch completed in 2.000000m 1.634180s
valid Loss: 0.8152 Acc: 0.7653

Epoch 5/9
----------
Epoch completed in 1.000000m 44.981900s
train Loss: 1.1529 Acc: 0.6955
Epoch completed in 1.000000m 59.446574s
valid Loss: 0.7941 Acc: 0.7800

Epoch 6/9
----------
Epoch completed in 1.000000m 44.666647s
train Loss: 1.1266 Acc: 0.6973
Epoch completed in 1.000000m 58.807862s
valid Loss: 0.7833 Acc: 0.7922

Epoch 7/9
----------
Epoch completed in 1.000000m 47.872149s
train Loss: 1.1305 Acc: 0.7042
Epoch completed in 2.000000m 2.699982s
valid Loss: 0.7918 Acc: 0.7775

Epoch 8/9
----------
Epoch completed in 1.000000m 48.224495s
train Loss: 1.0886 Acc: 0.7155
Epoch completed in 2.000000m 3.085865s
valid Loss: 0.7884 Acc: 0.7763

Epoch 9/9
----------
Epoch completed in 1.000000m 48.588541s
train Loss: 1.1139 Acc: 0.7021
Epoch completed in 2.000000m 3.330334s
valid Loss: 0.7884 Acc: 0.7775

Training complete in 31m 1s
Best val Acc: 0.792176
```

So the results are up. We can already see the saturation of accuracy at 0.7922 at epoch 7 of 10 (note that we started at epoch 0). What's next would be to finish first the rest of the notebook to make sure its working then we can continue working on improving the network.

Okay, so now the notebook is complete.

```python
'''
Here are the results.
Setup was: LR = 0.001, LR_scheduler was ReduceLROnPlateau, initial_model = vgg19_bn, 10 epochs, Batchsize = 64
Noticable improvements are:
Speed, it is now taking 3 mins per epoch possibly due to the Batch Size increase. The cloud service can take it so might as well use it.
Added RandomRotation to 360 degrees
'''

cuda
Epoch 0/9
----------
Epoch completed in 3.000000m 46.058207s
train Loss: 2.4487 Acc: 0.3773
Epoch completed in 0.000000m 27.437752s
valid Loss: 1.0728 Acc: 0.7372

Epoch 1/9
----------
Epoch completed in 3.000000m 45.175089s
train Loss: 2.0896 Acc: 0.4559
Epoch completed in 0.000000m 26.954424s
valid Loss: 0.9121 Acc: 0.7518

Epoch 2/9
----------
Epoch completed in 3.000000m 42.353104s
train Loss: 1.9727 Acc: 0.4944
Epoch completed in 0.000000m 27.458014s
valid Loss: 0.8118 Acc: 0.7702

Epoch 3/9
----------
Epoch completed in 3.000000m 46.368856s
train Loss: 1.8435 Acc: 0.5321
Epoch completed in 0.000000m 27.573256s
valid Loss: 0.7203 Acc: 0.8142

Epoch 4/9
----------
Epoch completed in 3.000000m 45.971057s
train Loss: 1.7503 Acc: 0.5525
Epoch completed in 0.000000m 27.610880s
valid Loss: 0.6261 Acc: 0.8484

Epoch 5/9
----------
Epoch completed in 3.000000m 45.741116s
train Loss: 1.7224 Acc: 0.5687
Epoch completed in 0.000000m 27.494761s
valid Loss: 0.7109 Acc: 0.8105

Epoch 6/9
----------
Epoch completed in 3.000000m 43.247956s
train Loss: 1.7163 Acc: 0.5704
Epoch completed in 0.000000m 26.951281s
valid Loss: 0.6066 Acc: 0.8594

Epoch 7/9
----------
Epoch completed in 3.000000m 43.880978s
train Loss: 1.6468 Acc: 0.5836
Epoch completed in 0.000000m 27.595762s
valid Loss: 0.5902 Acc: 0.8496

Epoch 8/9
----------
Epoch completed in 3.000000m 45.816934s
train Loss: 1.6005 Acc: 0.5955
Epoch completed in 0.000000m 27.279377s
valid Loss: 0.5809 Acc: 0.8570

Epoch 9/9
----------
Epoch completed in 3.000000m 45.665560s
train Loss: 1.6109 Acc: 0.5962
Epoch completed in 0.000000m 27.315036s
valid Loss: 0.5369 Acc: 0.8631

Training complete in 42m 4s
Best val Acc: 0.863081


```

Below is the screen shot of the utilization we have for the session. We have 3.06GB of RAM used, we could theoretically use a bigger batch size. The GPU is at 6.55GB with a max memory of 11GB possible.

I am planning on increasing the batch size further to 128 and push the epochs higher. :smirk:

<p align="center"><img src='.\Images\Colab-UTIL-VGG19_BN.png' width=800px></p>

```python
'''
Here are the results.
Setup was: LR = 0.001, LR_scheduler was ReduceLROnPlateau, initial_model = vgg19_bn, 15 epochs, Batchsize = 128

'''

cuda
Epoch 0/14
-*--*--*--*--*--*--*--*--*--*-
Epoch completed in 3.000000m 48.554248s
train Loss: 1.6426 Acc: 0.6064
Epoch completed in 0.000000m 27.318775s
valid Loss: 0.5432 Acc: 0.8557

Epoch 1/14
-*--*--*--*--*--*--*--*--*--*-
Epoch completed in 3.000000m 44.743166s
train Loss: 1.6360 Acc: 0.6103
Epoch completed in 0.000000m 28.019140s
valid Loss: 0.5955 Acc: 0.8606

Epoch 2/14
-*--*--*--*--*--*--*--*--*--*-
Epoch completed in 3.000000m 49.990680s
train Loss: 1.6658 Acc: 0.6036
Epoch completed in 0.000000m 28.249335s
valid Loss: 0.5436 Acc: 0.8729

Epoch 3/14
-*--*--*--*--*--*--*--*--*--*-
Epoch completed in 3.000000m 50.346915s
train Loss: 1.6462 Acc: 0.6061
Epoch completed in 0.000000m 28.010622s
valid Loss: 0.5786 Acc: 0.8619

Epoch 4/14
-*--*--*--*--*--*--*--*--*--*-
Epoch completed in 3.000000m 50.423357s
train Loss: 1.6258 Acc: 0.6134
Epoch completed in 0.000000m 28.069263s
valid Loss: 0.5805 Acc: 0.8557

Epoch 5/14
-*--*--*--*--*--*--*--*--*--*-
Epoch completed in 3.000000m 46.965881s
train Loss: 1.6364 Acc: 0.6151
Epoch completed in 0.000000m 27.376075s
valid Loss: 0.5519 Acc: 0.8778

Epoch 6/14
-*--*--*--*--*--*--*--*--*--*-
Epoch completed in 3.000000m 47.479331s
train Loss: 1.5779 Acc: 0.6247
Epoch completed in 0.000000m 28.113638s
valid Loss: 0.5504 Acc: 0.8729

Epoch 7/14
-*--*--*--*--*--*--*--*--*--*-
Epoch completed in 3.000000m 51.070040s
train Loss: 1.6519 Acc: 0.6082
Epoch completed in 0.000000m 28.155762s
valid Loss: 0.6243 Acc: 0.8716

Epoch 8/14
-*--*--*--*--*--*--*--*--*--*-
Epoch completed in 3.000000m 50.553549s
train Loss: 1.6031 Acc: 0.6224
Epoch completed in 0.000000m 28.016906s
valid Loss: 0.5203 Acc: 0.8888

Epoch 9/14
-*--*--*--*--*--*--*--*--*--*-
Epoch completed in 3.000000m 48.361469s
train Loss: 1.6074 Acc: 0.6207
Epoch completed in 0.000000m 27.537701s
valid Loss: 0.5419 Acc: 0.8680

Epoch 10/14
-*--*--*--*--*--*--*--*--*--*-
Epoch completed in 3.000000m 46.876984s
train Loss: 1.5788 Acc: 0.6276
Epoch completed in 0.000000m 28.044666s
valid Loss: 0.5595 Acc: 0.8826

Epoch 11/14
-*--*--*--*--*--*--*--*--*--*-
Epoch completed in 3.000000m 50.721860s
train Loss: 1.6030 Acc: 0.6239
Epoch completed in 0.000000m 28.022674s
valid Loss: 0.5975 Acc: 0.8667

Epoch 12/14
-*--*--*--*--*--*--*--*--*--*-
Epoch completed in 3.000000m 50.219541s
train Loss: 1.5591 Acc: 0.6352
Epoch completed in 0.000000m 28.055629s
valid Loss: 0.5514 Acc: 0.8680

Epoch 13/14
-*--*--*--*--*--*--*--*--*--*-
Epoch completed in 3.000000m 45.440233s
train Loss: 1.5317 Acc: 0.6285
Epoch completed in 0.000000m 27.294172s
valid Loss: 0.5530 Acc: 0.8778

Epoch 14/14
-*--*--*--*--*--*--*--*--*--*-
Epoch completed in 3.000000m 49.172531s
train Loss: 1.5443 Acc: 0.6320
Epoch completed in 0.000000m 28.048302s
valid Loss: 0.4972 Acc: 0.8961

Training complete in 64m 9s
Best val Acc: 0.896088
[(0, tensor(0.8557, device='cuda:0', dtype=torch.float64)), (1, tensor(0.8606, device='cuda:0', dtype=torch.float64)), (2, tensor(0.8729, device='cuda:0', dtype=torch.float64)), (3, tensor(0.8619, device='cuda:0', dtype=torch.float64)), (4, tensor(0.8557, device='cuda:0', dtype=torch.float64)), (5, tensor(0.8778, device='cuda:0', dtype=torch.float64)), (6, tensor(0.8729, device='cuda:0', dtype=torch.float64)), (7, tensor(0.8716, device='cuda:0', dtype=torch.float64)), (8, tensor(0.8888, device='cuda:0', dtype=torch.float64)), (9, tensor(0.8680, device='cuda:0', dtype=torch.float64)), (10, tensor(0.8826, device='cuda:0', dtype=torch.float64)), (11, tensor(0.8667, device='cuda:0', dtype=torch.float64)), (12, tensor(0.8680, device='cuda:0', dtype=torch.float64)), (13, tensor(0.8778, device='cuda:0', dtype=torch.float64)), (14, tensor(0.8961, device='cuda:0', dtype=torch.float64))]


```

So the speed improvement earlier was not due to the batch size. It would look like we are now saturated in terms of Accuracy. This is possibly the best we can do without data augmentation. We are just above the 1hr mark in terms of the training I guess this is it. The next improvements would be in terms of the data augmentation and balancing.

<p align="center"><img src='.\Images\Validation_Image-VGG19_BN(Stemless Gentian).png' width=500px></p>

Evidence of under representation effects.

<p align="center"><img src='.\Images\Validation_Image-VGG19_BN(Poinsettia).png' width=500px></p>

Note: File name is saved as `model_file_name = 'classifier_vgg19_bn_best.pth'. Also, we have a problem, the file size of VGG is severely limiting us, a path file for the model is 494MB when downloaded. That is a big file.

```python
'''
Found these on:
https://discuss.pytorch.org/t/balanced-sampling-between-classes-with-torchvision-dataloader/2703/2
https://discuss.pytorch.org/t/how-to-prevent-overfitting/1902/9
https://github.com/ufoym/imbalanced-dataset-sampler/blob/master/examples/imagefolder.ipynb
'''

def make_weights_for_balanced_classes(images, nclasses):
    count = [0] * nclasses
    for item in images:
        count[item[1]] += 1
    weight_per_class = [0.] * nclasses
    N = float(sum(count))
    for i in range(nclasses):
        weight_per_class[i] = N/float(count[i])
    weight = [0] * len(images)
    for idx, val in enumerate(images):
        weight[idx] = weight_per_class[val[1]]
    return weight

dataset_train = datasets.ImageFolder(traindir)
# For unbalanced dataset we create a weighted sampler
weights = make_weights_for_balanced_classes(dataset_train.imgs, len(dataset_train.classes))
weights = torch.DoubleTensor(weights)
sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))
train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle = True,sampler = sampler, num_workers=args.workers, pin_memory=True)
```

```python
'''
Here are the results.
Setup was: LR = 0.001, LR_scheduler was ReduceLROnPlateau, initial_model = densenet169, 15 epochs, Batchsize = 64
'''
cuda
Epoch 0/14
-*--*--*--*--*--*--*--*--*--*-
Epoch completed in 2.000000m 24.407018s
train Loss: 4.0518 Acc: 0.1114
Epoch completed in 0.000000m 18.858923s
valid Loss: 2.5880 Acc: 0.3619

Epoch 1/14
-*--*--*--*--*--*--*--*--*--*-
Epoch completed in 2.000000m 20.388797s
train Loss: 2.5784 Acc: 0.3494
Epoch completed in 0.000000m 18.746554s
valid Loss: 1.3678 Acc: 0.6161

Epoch 2/14
-*--*--*--*--*--*--*--*--*--*-
Epoch completed in 2.000000m 21.610583s
train Loss: 1.9339 Acc: 0.4748
Epoch completed in 0.000000m 18.397746s
valid Loss: 0.9029 Acc: 0.7592

Epoch 3/14
-*--*--*--*--*--*--*--*--*--*-
Epoch completed in 2.000000m 21.945431s
train Loss: 1.6675 Acc: 0.5449
Epoch completed in 0.000000m 19.228431s
valid Loss: 0.7057 Acc: 0.8154

Epoch 4/14
-*--*--*--*--*--*--*--*--*--*-
Epoch completed in 2.000000m 23.849681s
train Loss: 1.4893 Acc: 0.5871
Epoch completed in 0.000000m 19.253587s
valid Loss: 0.5992 Acc: 0.8386

Epoch 5/14
-*--*--*--*--*--*--*--*--*--*-
Epoch completed in 2.000000m 24.447653s
train Loss: 1.3423 Acc: 0.6320
Epoch completed in 0.000000m 19.101235s
valid Loss: 0.5301 Acc: 0.8496

Epoch 6/14
-*--*--*--*--*--*--*--*--*--*-
Epoch completed in 2.000000m 24.337290s
train Loss: 1.2626 Acc: 0.6534
Epoch completed in 0.000000m 19.072001s
valid Loss: 0.4738 Acc: 0.8729

Epoch 7/14
-*--*--*--*--*--*--*--*--*--*-
Epoch completed in 2.000000m 25.040146s
train Loss: 1.2344 Acc: 0.6569
Epoch completed in 0.000000m 19.275203s
valid Loss: 0.4263 Acc: 0.8985

Epoch 8/14
-*--*--*--*--*--*--*--*--*--*-
Epoch completed in 2.000000m 26.248204s
train Loss: 1.1774 Acc: 0.6749
Epoch completed in 0.000000m 19.331133s
valid Loss: 0.3587 Acc: 0.9059

Epoch 9/14
-*--*--*--*--*--*--*--*--*--*-
Epoch completed in 2.000000m 22.067156s
train Loss: 1.0971 Acc: 0.7013
Epoch completed in 0.000000m 18.883429s
valid Loss: 0.3644 Acc: 0.9071

Epoch 10/14
-*--*--*--*--*--*--*--*--*--*-
Epoch completed in 2.000000m 23.130489s
train Loss: 1.0878 Acc: 0.7005
Epoch completed in 0.000000m 18.699926s
valid Loss: 0.3547 Acc: 0.9010

Epoch 11/14
-*--*--*--*--*--*--*--*--*--*-
Epoch completed in 2.000000m 23.717617s
train Loss: 1.0312 Acc: 0.7172
Epoch completed in 0.000000m 19.399459s
valid Loss: 0.3473 Acc: 0.9108

Epoch 12/14
-*--*--*--*--*--*--*--*--*--*-
Epoch completed in 2.000000m 26.140493s
train Loss: 1.0104 Acc: 0.7256
Epoch completed in 0.000000m 19.259526s
valid Loss: 0.3489 Acc: 0.9034

Epoch 13/14
-*--*--*--*--*--*--*--*--*--*-
Epoch completed in 2.000000m 26.161979s
train Loss: 1.0299 Acc: 0.7228
Epoch completed in 0.000000m 19.334597s
valid Loss: 0.3410 Acc: 0.9120

Epoch 14/14
-*--*--*--*--*--*--*--*--*--*-
Epoch completed in 2.000000m 26.249305s
train Loss: 0.9726 Acc: 0.7335
Epoch completed in 0.000000m 19.357714s
valid Loss: 0.3090 Acc: 0.9181

Training complete in 40m 47s
Best val Acc: 0.918093
[(0, tensor(0.3619, device='cuda:0', dtype=torch.float64)), (1, tensor(0.6161, device='cuda:0', dtype=torch.float64)), (2, tensor(0.7592, device='cuda:0', dtype=torch.float64)), (3, tensor(0.8154, device='cuda:0', dtype=torch.float64)), (4, tensor(0.8386, device='cuda:0', dtype=torch.float64)), (5, tensor(0.8496, device='cuda:0', dtype=torch.float64)), (6, tensor(0.8729, device='cuda:0', dtype=torch.float64)), (7, tensor(0.8985, device='cuda:0', dtype=torch.float64)), (8, tensor(0.9059, device='cuda:0', dtype=torch.float64)), (9, tensor(0.9071, device='cuda:0', dtype=torch.float64)), (10, tensor(0.9010, device='cuda:0', dtype=torch.float64)), (11, tensor(0.9108, device='cuda:0', dtype=torch.float64)), (12, tensor(0.9034, device='cuda:0', dtype=torch.float64)), (13, tensor(0.9120, device='cuda:0', dtype=torch.float64)), (14, tensor(0.9181, device='cuda:0', dtype=torch.float64))]
```

So now we are looking at the Densenet169 results. The results in the accuracy are higher which is good. But without the confusion matrix its hard to tell which part its having issues with. The marked improvement is the speed. Also, the size is now 54MB which is more manageable. This is good. The next step is to make the confusion matrix and the random sampling and balancing datasets.

<p align="center"><img src='.\Images\Validation_Image-Densenet169(Stemless Gentian).png' width=500px></p>

<p align="center"><img src='.\Images\Validation_Image-Densenet169(Poinsettia).png' width=500px></p>

:gem: Additional guide from [Keras on Pre-trained models](https://keras.io/applications/)

## December 18, 2018 - Day 39 of the Challenge

So I am now back home and have set up my schedules for the holiday. First up for the non-work and non-family related activities is to complete the challenge. For today, I am checking on changing the LR_Scheduler from the plateau to Cosine annealing. Just checking if we get better results and speed.

Next step has always been to deal with the imbalanced data with weights initialization during loading. This needs more reading on how to implement.

```python
'''
Here are the results.
Setup was: LR = 0.001, LR_scheduler = CosineAnnealingLR(optimizer, 5, eta_min=0.0001, last_epoch=-1), initial_model = densenet169, 15 epochs, Batchsize = 64
'''

# NOTE: Failed to run. This was not converging so I had to stop it. Baseline is just 2-3 mins per epoch based on previous implementation. I let it run for 10 mins before I interrupted the run.

'''
# Criteria NLLLoss which is recommended with Softmax final layer
criteria = nn.NLLLoss()
# Observe that all parameters are being optimized
optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)
# Decay LR by a factor of 0.1 every 4 epochs
sched = lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.1)
# Number of epochs
eps=15

# TODO: Define our classifier that will be replacing the last layer in VGG19.
classifier = nn.Sequential(
    OrderedDict([
        ('fc1',nn.Linear(1664,832)),
        ('relu_1',nn.ReLU()),
        ('drop_1',nn.Dropout(p=0.5)),
        ('fc2',nn.Linear(832,256)),
        ('relu_2',nn.ReLU()),
        ('drop_2',nn.Dropout(p=0.5)),
        ('fc3',nn.Linear(256,102)),
        ('output',nn.LogSoftmax(dim = 1))
    ]))
# Transfer the classifier results
model.classifier = classifier
'''

cuda
Epoch 0/14
----------
train epoch completed in 2.000000m 15.521907s
train Loss: 1.4853 Acc: 0.5940
valid epoch completed in 2.000000m 33.571836s
valid Loss: 0.6899 Acc: 0.8240

Epoch 1/14
----------
train epoch completed in 2.000000m 15.721620s
train Loss: 1.3806 Acc: 0.6212
valid epoch completed in 2.000000m 33.740467s
valid Loss: 0.6240 Acc: 0.8582

Epoch 2/14
----------
train epoch completed in 2.000000m 15.844700s
train Loss: 1.2923 Acc: 0.6438
valid epoch completed in 2.000000m 33.840051s
valid Loss: 0.5662 Acc: 0.8594

Epoch 3/14
----------
train epoch completed in 2.000000m 15.681037s
train Loss: 1.2297 Acc: 0.6589
valid epoch completed in 2.000000m 33.712121s
valid Loss: 0.5207 Acc: 0.8680

Epoch 4/14
----------
train epoch completed in 2.000000m 15.632705s
train Loss: 1.0636 Acc: 0.7117
valid epoch completed in 2.000000m 33.643717s
valid Loss: 0.4406 Acc: 0.8875

Epoch 5/14
----------
train epoch completed in 2.000000m 15.663321s
train Loss: 0.9841 Acc: 0.7271
valid epoch completed in 2.000000m 33.618531s
valid Loss: 0.4310 Acc: 0.8851

Epoch 6/14
----------
train epoch completed in 2.000000m 15.379978s
train Loss: 0.9456 Acc: 0.7337
valid epoch completed in 2.000000m 33.347549s
valid Loss: 0.4223 Acc: 0.8973

Epoch 7/14
----------
train epoch completed in 2.000000m 15.419425s
train Loss: 0.9903 Acc: 0.7274
valid epoch completed in 2.000000m 33.390536s
valid Loss: 0.4082 Acc: 0.8985

Epoch 8/14
----------
train epoch completed in 2.000000m 15.504078s
train Loss: 0.9258 Acc: 0.7408
valid epoch completed in 2.000000m 33.512190s
valid Loss: 0.4087 Acc: 0.8912

Epoch 9/14
----------
train epoch completed in 2.000000m 15.572731s
train Loss: 0.9376 Acc: 0.7392
valid epoch completed in 2.000000m 33.562267s
valid Loss: 0.4098 Acc: 0.8912

Epoch 10/14
----------
train epoch completed in 2.000000m 15.705945s
train Loss: 0.9370 Acc: 0.7437
valid epoch completed in 2.000000m 33.676492s
valid Loss: 0.4098 Acc: 0.8998

Epoch 11/14
----------
train epoch completed in 2.000000m 15.553149s
train Loss: 0.9427 Acc: 0.7418
valid epoch completed in 2.000000m 33.542307s
valid Loss: 0.4039 Acc: 0.8949

Epoch 12/14
----------
train epoch completed in 2.000000m 15.767509s
train Loss: 0.9314 Acc: 0.7440
valid epoch completed in 2.000000m 33.803519s
valid Loss: 0.4091 Acc: 0.8888

Epoch 13/14
----------
train epoch completed in 2.000000m 15.360914s
train Loss: 0.9496 Acc: 0.7346
valid epoch completed in 2.000000m 33.314968s
valid Loss: 0.4088 Acc: 0.8985

Epoch 14/14
----------
train epoch completed in 2.000000m 15.313187s
train Loss: 0.9426 Acc: 0.7357
valid epoch completed in 2.000000m 33.208476s
valid Loss: 0.4045 Acc: 0.8973

Training complete in 38m 24s
Best val Acc: 0.899756

```

Leaky ReLU is leading.

```python
'''
Setup:

# Criteria NLLLoss which is recommended with Softmax final layer
criteria = nn.NLLLoss()
# Observe that all parameters are being optimized
optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)
# Decay LR by a factor of 0.1 every 4 epochs
# sched = lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.1)
sched = lr_scheduler.MultiStepLR(optimizer, milestones=[5,10], gamma=0.1)
# Number of epochs
eps=15


# TODO: Define our classifier that will be replacing the last layer in VGG19.
classifier = nn.Sequential(
    OrderedDict([
        ('fc1',nn.Linear(1664,832)),
        ('relu_1',nn.ReLU()),
        ('drop_1',nn.Dropout(p=0.5)),
        ('fc2',nn.Linear(832,256)),
        ('relu_2',nn.ReLU()),
        ('drop_2',nn.Dropout(p=0.5)),
        ('fc3',nn.Linear(256,102)),
        ('output',nn.LogSoftmax(dim = 1))
    ]))
# Transfer the classifier results
model.classifier = classifier

Saved as: classifier_densenet169_V2.pth
'''


cuda
Epoch 0/14
----------
train epoch completed in 27.000000m 50.823827s
train Loss: 4.2127 Acc: 0.0968
valid epoch completed in 32.000000m 1.857351s
valid Loss: 3.0348 Acc: 0.3509

Epoch 1/14
----------
train epoch completed in 2.000000m 16.945112s
train Loss: 2.7928 Acc: 0.3127
valid epoch completed in 2.000000m 35.400791s
valid Loss: 1.5916 Acc: 0.5856

Epoch 2/14
----------
train epoch completed in 2.000000m 17.341003s
train Loss: 2.0218 Acc: 0.4664
valid epoch completed in 2.000000m 35.742949s
valid Loss: 1.0676 Acc: 0.7372

Epoch 3/14
----------
train epoch completed in 2.000000m 16.459388s
train Loss: 1.6944 Acc: 0.5449
valid epoch completed in 2.000000m 34.693902s
valid Loss: 0.8405 Acc: 0.7812

Epoch 4/14
----------
train epoch completed in 2.000000m 17.231413s
train Loss: 1.4676 Acc: 0.5934
valid epoch completed in 2.000000m 35.613135s
valid Loss: 0.6584 Acc: 0.8325

Epoch 5/14
----------
train epoch completed in 2.000000m 16.795884s
train Loss: 1.2269 Acc: 0.6586
valid epoch completed in 2.000000m 35.062843s
valid Loss: 0.5661 Acc: 0.8582

Epoch 6/14
----------
train epoch completed in 2.000000m 16.483428s
train Loss: 1.2152 Acc: 0.6751
valid epoch completed in 2.000000m 35.016291s
valid Loss: 0.5456 Acc: 0.8655

Epoch 7/14
----------
train epoch completed in 2.000000m 16.653298s
train Loss: 1.1587 Acc: 0.6833
valid epoch completed in 2.000000m 34.980344s
valid Loss: 0.5370 Acc: 0.8606

Epoch 8/14
----------
train epoch completed in 2.000000m 16.881268s
train Loss: 1.1406 Acc: 0.6835
valid epoch completed in 2.000000m 35.196797s
valid Loss: 0.5188 Acc: 0.8680

Epoch 9/14
----------
train epoch completed in 2.000000m 17.073467s
train Loss: 1.1310 Acc: 0.6903
valid epoch completed in 2.000000m 35.421782s
valid Loss: 0.4997 Acc: 0.8741

Epoch 10/14
----------
train epoch completed in 2.000000m 16.765958s
train Loss: 1.1389 Acc: 0.6917
valid epoch completed in 2.000000m 35.108592s
valid Loss: 0.5080 Acc: 0.8704

Epoch 11/14
----------
train epoch completed in 2.000000m 16.707246s
train Loss: 1.1376 Acc: 0.6845
valid epoch completed in 2.000000m 34.987313s
valid Loss: 0.4987 Acc: 0.8729

Epoch 12/14
----------
train epoch completed in 2.000000m 16.688344s
train Loss: 1.1085 Acc: 0.6981
valid epoch completed in 2.000000m 35.346166s
valid Loss: 0.4955 Acc: 0.8741

Epoch 13/14
----------
train epoch completed in 2.000000m 16.558425s
train Loss: 1.0919 Acc: 0.7015
valid epoch completed in 2.000000m 34.846010s
valid Loss: 0.4938 Acc: 0.8753

Epoch 14/14
----------
train epoch completed in 2.000000m 16.554662s
train Loss: 1.0972 Acc: 0.6961
valid epoch completed in 2.000000m 35.296205s
valid Loss: 0.4932 Acc: 0.8729

Training complete in 68m 16s
Best val Acc: 0.875306


```

Quick Segway to the results from V1 Classifier.

```python
'''
classifier_densenet169_V1.pth
'''

Batch accuracy (Size 32): 0.96875
Batch accuracy (Size 32): 0.96875
Batch accuracy (Size 32): 0.84375
Batch accuracy (Size 32): 0.84375
Batch accuracy (Size 32): 1.0
Batch accuracy (Size 32): 0.9375
Batch accuracy (Size 32): 0.96875
Batch accuracy (Size 32): 1.0
Batch accuracy (Size 32): 0.96875
Batch accuracy (Size 32): 0.90625
Batch accuracy (Size 32): 0.90625
Batch accuracy (Size 32): 1.0
Batch accuracy (Size 32): 0.90625
Batch accuracy (Size 32): 0.9375
Batch accuracy (Size 32): 0.90625
Batch accuracy (Size 32): 0.90625
Batch accuracy (Size 32): 0.8125
Batch accuracy (Size 32): 0.875
Batch accuracy (Size 32): 0.90625
Batch accuracy (Size 32): 0.875
Batch accuracy (Size 32): 0.875
Batch accuracy (Size 32): 0.8125
Batch accuracy (Size 32): 0.84375
Batch accuracy (Size 32): 0.96875
Batch accuracy (Size 32): 0.96875
Batch accuracy (Size 32): 1.0
Mean accuracy: 0.9194711446762085

0.91947114
```

```python
'''
classifier_densenet169_V2.pth
'''
Batch accuracy (Size 32): 0.90625
Batch accuracy (Size 32): 0.90625
Batch accuracy (Size 32): 0.90625
Batch accuracy (Size 32): 0.875
Batch accuracy (Size 32): 0.84375
Batch accuracy (Size 32): 0.84375
Batch accuracy (Size 32): 0.9375
Batch accuracy (Size 32): 0.8125
Batch accuracy (Size 32): 0.90625
Batch accuracy (Size 32): 0.78125
Batch accuracy (Size 32): 0.78125
Batch accuracy (Size 32): 0.84375
Batch accuracy (Size 32): 0.9375
Batch accuracy (Size 32): 0.875
Batch accuracy (Size 32): 0.90625
Batch accuracy (Size 32): 0.875
Batch accuracy (Size 32): 0.90625
Batch accuracy (Size 32): 0.9375
Batch accuracy (Size 32): 1.0
Batch accuracy (Size 32): 0.90625
Batch accuracy (Size 32): 0.875
Batch accuracy (Size 32): 0.8125
Batch accuracy (Size 32): 0.90625
Batch accuracy (Size 32): 0.90625
Batch accuracy (Size 32): 0.90625
Batch accuracy (Size 32): 0.8333333134651184
Mean accuracy: 0.8818109035491943

0.8818109
```

```python
'''
classifier_densenet169_V3.pth
Changes: SGD optim used
optimizer = optim.SGD(model.classifier.parameters(), lr=0.01, momentum = 0.9)
'''
Batch accuracy (Size 32): 0.90625
Batch accuracy (Size 32): 0.84375
Batch accuracy (Size 32): 0.9375
Batch accuracy (Size 32): 0.90625
Batch accuracy (Size 32): 0.84375
Batch accuracy (Size 32): 0.90625
Batch accuracy (Size 32): 0.78125
Batch accuracy (Size 32): 0.875
Batch accuracy (Size 32): 0.96875
Batch accuracy (Size 32): 0.9375
Batch accuracy (Size 32): 0.90625
Batch accuracy (Size 32): 0.90625
Batch accuracy (Size 32): 0.84375
Batch accuracy (Size 32): 0.9375
Batch accuracy (Size 32): 0.90625
Batch accuracy (Size 32): 0.9375
Batch accuracy (Size 32): 0.90625
Batch accuracy (Size 32): 0.875
Batch accuracy (Size 32): 0.90625
Batch accuracy (Size 32): 0.875
Batch accuracy (Size 32): 0.9375
Batch accuracy (Size 32): 0.90625
Batch accuracy (Size 32): 0.90625
Batch accuracy (Size 32): 0.96875
Batch accuracy (Size 32): 0.96875
Batch accuracy (Size 32): 0.944444477558136
Mean accuracy: 0.9053151607513428

0.90531516
```

```python
'''
classifier_densenet169_V4.pth
Changes: SGD optim used
optimizer = optim.SGD(model.classifier.parameters(), lr=0.01, momentum = 0.9)
LR_Sched = Reduce on platue (?) Check
'''
Batch accuracy (Size 32): 0.90625
Batch accuracy (Size 32): 0.84375
Batch accuracy (Size 32): 0.9375
Batch accuracy (Size 32): 0.90625
Batch accuracy (Size 32): 0.84375
Batch accuracy (Size 32): 0.90625
Batch accuracy (Size 32): 0.78125
Batch accuracy (Size 32): 0.875
Batch accuracy (Size 32): 0.96875
Batch accuracy (Size 32): 0.9375
Batch accuracy (Size 32): 0.90625
Batch accuracy (Size 32): 0.90625
Batch accuracy (Size 32): 0.84375
Batch accuracy (Size 32): 0.9375
Batch accuracy (Size 32): 0.90625
Batch accuracy (Size 32): 0.9375
Batch accuracy (Size 32): 0.90625
Batch accuracy (Size 32): 0.875
Batch accuracy (Size 32): 0.90625
Batch accuracy (Size 32): 0.875
Batch accuracy (Size 32): 0.9375
Batch accuracy (Size 32): 0.90625
Batch accuracy (Size 32): 0.90625
Batch accuracy (Size 32): 0.96875
Batch accuracy (Size 32): 0.96875
Batch accuracy (Size 32): 0.944444477558136
Mean accuracy: 0.9053151607513428

0.90531516
```

I need to study [Finetuning in PyTorch](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html). It could improve the results possibly. Also I have figured out why yesterday's run did not run completely, the optimizer/scheduler.step part was on the wrong config.

```python
'''study this:https://gist.github.com/avijit9/1c7eebf124a02a555f7626a0fbcd04a5
'''
# pdb.set_trace()

    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD([
                {'params': model.conv1.parameters()},
                {'params': model.bn1.parameters()},
                {'params': model.relu.parameters()},
                {'params': model.maxpool.parameters()},
                {'params': model.layer1.parameters()},
                {'params': model.layer2.parameters()},
                {'params': model.layer3.parameters()},
                {'params': model.layer4.parameters()},
                {'params': model.avgpool.parameters()},
                {'params': model.fc.parameters(), 'lr': opt.lr}
            ], lr=opt.lr*0.1, momentum=0.9)
    #optimizer = torch.optim.SGD(model.parameters(), opt.lr, momentum = opt.momentum, weight_decay = opt.weight_decay)
#pdb.set_trace()
```

## December 28, 2018 - Day 49 of the Challenge

Today is the resumption of the challenge. I was on Christmas break and have just found the time again to resume training and improving the model. :hocho: :bomb:

This is cutting it close but it is still on. Also, I have a mentee in the program. She is also an Electronics/Electrical engineering major from Malaysia. Our goal is for her to be able to cram as much lessons as possible and then join in on the challenge. She does not seem to have questions so my approach would be to motivate her and check up on her progress. :wink: Its great that I have a check on my own progress as well.

```python
'''
Changed the transforms, added is the more appropriate term:
Added color jitter to randomly apply variations in the color. We would want the shape to be the basis YES?
Tried adding affine transformation as well.


        transforms.ColorJitter(brightness=10, contrast=10, saturation=10, hue=10),
        transforms.RandomAffine(degrees=360.0, translate=None, scale=(0.5,2.0), shear=90.0, resample=False, fillcolor=0),

Reduced the dropout rate to 0.3 was 0.5


# TODO: Define our classifier that will be replacing the last layer in VGG19.
classifier = nn.Sequential(
    OrderedDict([
        ('fc1',nn.Linear(1664,832)),
        ('relu_1',nn.ReLU()),
        ('drop_1',nn.Dropout(p=0.3)),
        ('fc2',nn.Linear(832,256)),
        ('relu_2',nn.ReLU()),
        ('drop_2',nn.Dropout(p=0.3)),
        ('fc3',nn.Linear(256,102)),
        ('output',nn.LogSoftmax(dim = 1))
    ]))
# Transfer the classifier results
model.classifier = classifier
'''
# pdb.set_trace()

    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD([
                {'params': model.conv1.parameters()},
                {'params': model.bn1.parameters()},
                {'params': model.relu.parameters()},
                {'params': model.maxpool.parameters()},
                {'params': model.layer1.parameters()},
                {'params': model.layer2.parameters()},
                {'params': model.layer3.parameters()},
                {'params': model.layer4.parameters()},
                {'params': model.avgpool.parameters()},
                {'params': model.fc.parameters(), 'lr': opt.lr}
            ], lr=opt.lr*0.1, momentum=0.9)
    #optimizer = torch.optim.SGD(model.parameters(), opt.lr, momentum = opt.momentum, weight_decay = opt.weight_decay)
#pdb.set_trace()

Batch accuracy (Size 32): 0.15625
Batch accuracy (Size 32): 0.125
Batch accuracy (Size 32): 0.15625
Batch accuracy (Size 32): 0.09375
Batch accuracy (Size 32): 0.21875
Batch accuracy (Size 32): 0.1875
Batch accuracy (Size 32): 0.125
Batch accuracy (Size 32): 0.15625
Batch accuracy (Size 32): 0.28125
Batch accuracy (Size 32): 0.1875
Batch accuracy (Size 32): 0.21875
Batch accuracy (Size 32): 0.25
Batch accuracy (Size 32): 0.15625
Batch accuracy (Size 32): 0.21875
Batch accuracy (Size 32): 0.09375
Batch accuracy (Size 32): 0.25
Batch accuracy (Size 32): 0.0
Batch accuracy (Size 32): 0.09375
Batch accuracy (Size 32): 0.125
Batch accuracy (Size 32): 0.09375
Batch accuracy (Size 32): 0.15625
Batch accuracy (Size 32): 0.03125
Batch accuracy (Size 32): 0.21875
Batch accuracy (Size 32): 0.125
Batch accuracy (Size 32): 0.0625
Batch accuracy (Size 32): 0.2222222238779068
Mean accuracy: 0.15397970378398895

0.1539797
```

So the results are not that promising. The accuracy is low to begin with, possibly due the randomness of affine. Possibly turn of the affine for now.

I have removed random affine for the next run below. Off the bat the speed is really fast. I think I have figured out how transform works. I think the transforms happen at the very first epoch that is why using very intensive transforms are going to eat up the compute time at the very first epoch run.

```python
'''
removed the affine random transform. Retained the rest of the setup from above.
'''


Batch accuracy (Size 32): 0.5
Batch accuracy (Size 32): 0.53125
Batch accuracy (Size 32): 0.6875
Batch accuracy (Size 32): 0.625
Batch accuracy (Size 32): 0.59375
Batch accuracy (Size 32): 0.6875
Batch accuracy (Size 32): 0.53125
Batch accuracy (Size 32): 0.46875
Batch accuracy (Size 32): 0.53125
Batch accuracy (Size 32): 0.6875
Batch accuracy (Size 32): 0.59375
Batch accuracy (Size 32): 0.5
Batch accuracy (Size 32): 0.6875
Batch accuracy (Size 32): 0.625
Batch accuracy (Size 32): 0.625
Batch accuracy (Size 32): 0.625
Batch accuracy (Size 32): 0.5625
Batch accuracy (Size 32): 0.6875
Batch accuracy (Size 32): 0.53125
Batch accuracy (Size 32): 0.40625
Batch accuracy (Size 32): 0.5625
Batch accuracy (Size 32): 0.5625
Batch accuracy (Size 32): 0.625
Batch accuracy (Size 32): 0.5625
Batch accuracy (Size 32): 0.53125
Batch accuracy (Size 32): 0.5
Mean accuracy: 0.578125

0.578125
```

So it was affine random that caused the very low accuracy score earlier. Now the max we can get is 0.5. So the idea is now to move the dropout probability back to 0.5. And its looking like the training is still stuck. Tried 25 episodes in the code run below and still have 50% accuracy, possibly due to the smaller size of the data?

Will have to return to the earlier model which had 91% accuracy.

```python
'''
What did I change?
* I changed the episode count to 25 epochs to check if we can train it further, turns out we cannot with the color jitter. Its not going to matter how much we change and augment the data when our sampling is off so I need to fix it.

'''

# RESULTS: TEST

Batch accuracy (Size 32): 0.59375
Batch accuracy (Size 32): 0.625
Batch accuracy (Size 32): 0.59375
Batch accuracy (Size 32): 0.65625
Batch accuracy (Size 32): 0.59375
Batch accuracy (Size 32): 0.53125
Batch accuracy (Size 32): 0.71875
Batch accuracy (Size 32): 0.53125
Batch accuracy (Size 32): 0.6875
Batch accuracy (Size 32): 0.5625
Batch accuracy (Size 32): 0.625
Batch accuracy (Size 32): 0.5625
Batch accuracy (Size 32): 0.71875
Batch accuracy (Size 32): 0.71875
Batch accuracy (Size 32): 0.5
Batch accuracy (Size 32): 0.5
Batch accuracy (Size 32): 0.46875
Batch accuracy (Size 32): 0.40625
Batch accuracy (Size 32): 0.625
Batch accuracy (Size 32): 0.625
Batch accuracy (Size 32): 0.5
Batch accuracy (Size 32): 0.59375
Batch accuracy (Size 32): 0.59375
Batch accuracy (Size 32): 0.625
Batch accuracy (Size 32): 0.5625
Batch accuracy (Size 32): 0.5555555820465088
Mean accuracy: 0.5874732732772827

0.5874733
```

For reference here was what we were able to achieve before we did the changes today:
The accuracy was already high without the transforms. Actually, its not that we are transforming it that is causing the issue because theoretically it should add more robustness to our model. The problem is that the model was imbalanced to begin with causing the overtraining of other images compared to others which causes the issue (I think).

```python
Batch accuracy (Size 32): 0.90625
Batch accuracy (Size 32): 0.875
Batch accuracy (Size 32): 0.96875
Batch accuracy (Size 32): 0.875
Batch accuracy (Size 32): 0.96875
Batch accuracy (Size 32): 0.90625
Batch accuracy (Size 32): 0.875
Batch accuracy (Size 32): 0.96875
Batch accuracy (Size 32): 0.90625
Batch accuracy (Size 32): 0.84375
Batch accuracy (Size 32): 0.875
Batch accuracy (Size 32): 0.9375
Batch accuracy (Size 32): 0.875
Batch accuracy (Size 32): 0.90625
Batch accuracy (Size 32): 0.90625
Batch accuracy (Size 32): 0.96875
Batch accuracy (Size 32): 0.90625
Batch accuracy (Size 32): 0.90625
Batch accuracy (Size 32): 0.84375
Batch accuracy (Size 32): 0.96875
Batch accuracy (Size 32): 0.875
Batch accuracy (Size 32): 0.875
Batch accuracy (Size 32): 0.875
Batch accuracy (Size 32): 0.96875
Batch accuracy (Size 32): 0.96875
Batch accuracy (Size 32): 0.944444477558136
Mean accuracy: 0.9113247990608215

0.9113248
```

## December 29, 2018 - Day 50 of the Challenge

Goal for today is to figure out the sampler. But the first run would be to use the earlier model in V3 and then train it for more epochs to see if we can actually move past 91% in accuracy.

```python
'''
What did I do?
Return to the model for V3 and then train it for 30 episodes to see if it can move past the 91% accuracy.
'''

# RESULT

Epoch 29/29
-*--*--*--*--*--*--*--*--*--*-
Epoch completed in 2.000000m 14.571949s
train Loss: 0.6298 Acc: 0.8230
Epoch completed in 0.000000m 17.766753s
valid Loss: 0.2374 Acc: 0.9340

Training complete in 76m 5s
Best val Acc: 0.940098
[(0, tensor(0.6968, device='cuda:0', dtype=torch.float64)), (1, tensor(0.7592, device='cuda:0', dtype=torch.float64)), (2, tensor(0.8130, device='cuda:0', dtype=torch.float64)), (3, tensor(0.8496, device='cuda:0', dtype=torch.float64)), (4, tensor(0.8447, device='cuda:0', dtype=torch.float64)), (5, tensor(0.8729, device='cuda:0', dtype=torch.float64)), (6, tensor(0.8594, device='cuda:0', dtype=torch.float64)), (7, tensor(0.8680, device='cuda:0', dtype=torch.float64)), (8, tensor(0.8875, device='cuda:0', dtype=torch.float64)), (9, tensor(0.8900, device='cuda:0', dtype=torch.float64)), (10, tensor(0.8888, device='cuda:0', dtype=torch.float64)), (11, tensor(0.9022, device='cuda:0', dtype=torch.float64)), (12, tensor(0.9034, device='cuda:0', dtype=torch.float64)), (13, tensor(0.9059, device='cuda:0', dtype=torch.float64)), (14, tensor(0.9010, device='cuda:0', dtype=torch.float64)), (15, tensor(0.9193, device='cuda:0', dtype=torch.float64)), (16, tensor(0.9132, device='cuda:0', dtype=torch.float64)), (17, tensor(0.9242, device='cuda:0', dtype=torch.float64)), (18, tensor(0.9193, device='cuda:0', dtype=torch.float64)), (19, tensor(0.9328, device='cuda:0', dtype=torch.float64)), (20, tensor(0.9254, device='cuda:0', dtype=torch.float64)), (21, tensor(0.9303, device='cuda:0', dtype=torch.float64)), (22, tensor(0.9230, device='cuda:0', dtype=torch.float64)), (23, tensor(0.9401, device='cuda:0', dtype=torch.float64)), (24, tensor(0.9389, device='cuda:0', dtype=torch.float64)), (25, tensor(0.9291, device='cuda:0', dtype=torch.float64)), (26, tensor(0.9401, device='cuda:0', dtype=torch.float64)), (27, tensor(0.9377, device='cuda:0', dtype=torch.float64)), (28, tensor(0.9328, device='cuda:0', dtype=torch.float64)), (29, tensor(0.9340, device='cuda:0', dtype=torch.float64))]

Batch accuracy (Size 32): 0.96875
Batch accuracy (Size 32): 0.9375
Batch accuracy (Size 32): 0.96875
Batch accuracy (Size 32): 0.90625
Batch accuracy (Size 32): 0.96875
Batch accuracy (Size 32): 0.9375
Batch accuracy (Size 32): 0.96875
Batch accuracy (Size 32): 0.96875
Batch accuracy (Size 32): 0.875
Batch accuracy (Size 32): 0.96875
Batch accuracy (Size 32): 0.96875
Batch accuracy (Size 32): 0.90625
Batch accuracy (Size 32): 0.875
Batch accuracy (Size 32): 0.875
Batch accuracy (Size 32): 0.9375
Batch accuracy (Size 32): 1.0
Batch accuracy (Size 32): 1.0
Batch accuracy (Size 32): 0.96875
Batch accuracy (Size 32): 0.96875
Batch accuracy (Size 32): 0.90625
Batch accuracy (Size 32): 0.9375
Batch accuracy (Size 32): 0.84375
Batch accuracy (Size 32): 0.84375
Batch accuracy (Size 32): 0.96875
Batch accuracy (Size 32): 0.96875
Batch accuracy (Size 32): 0.8888888955116272
Mean accuracy: 0.9356303811073303

0.9356304

```
