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

Since it is obviously not working, I need a new approach.

[Verification OpenSource](https://github.com/GabrielePicco/deep-learning-flower-identifier)