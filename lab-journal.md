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