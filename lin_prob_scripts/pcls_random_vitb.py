from src.models.vision_transformer import vit_huge
from src import helper
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import torch.nn.functional as F
import time
from datetime import timedelta

# Initialize the ViT-H model with the specified patch size and resolution
# model = vit_huge(patch_size=4, num_classes=1000) # Adjust num_classes if needed

IMG_CROPSIZE = 150
NUM_CLASSES = 6
SAVE_PATH = 'classifiers/jepa_iic_classifier_locked_randomvitb'
LR = 0.001
# NUM_EPOCHS = 300
NUM_EPOCHS = 100
BATCH_SIZE = 256
# Define paths to datasets
train_data_path = 'datasets/intel-image-classification/train'
val_data_path = 'datasets/intel-image-classification/test'

# EMBED_DIMS=1024 # for ViT-large
# EMBED_DIMS=1280 # for ViT-huge
EMBED_DIMS=768 # for ViT-base



encoder, predictor = helper.init_model(device='cuda:0', 
                                       patch_size=15,
                                       model_name='vit_base',
                                       crop_size=IMG_CROPSIZE,
                                       pred_depth=12,
                                       pred_emb_dim=384)


load_encoder = False
if load_encoder: # In this file we perform a test, no loading takes place
  # Load the state dictionary from the file
  load_path = 'logs/tin_A100/jepa-latest.pth.tar'
  ckpt = torch.load(load_path, map_location=torch.device('cpu'))
  # state_dict = torch.load('/content/IN1K-vit.h.14-300e.pth.tar')
  pretrained_dict = ckpt['encoder']

  # -- loading encoder
  for k, v in pretrained_dict.items():
    encoder.state_dict()[k[len('module.'):]].copy_(v) 

# Print the layers/modules of the model for inspection
def print_model_layers(model, prefix=''):
  for name, module in model.named_children():
    if isinstance(module, torch.nn.Module):
      module_name = prefix + '.' + name if prefix else name
      print(module_name)
      print_model_layers(module, prefix=module_name)

# print_model_layers(encoder) # alright this works :)
# print('INFO Gogos - Printing the predictor\'s architecture.')
# print_model_layers(predictor) # 
# print('Done with predictor\'s architecture.')


class ClassifierHead(nn.Module):
  def __init__(self, input_size, num_classes):
    super(ClassifierHead, self).__init__()
    hidden_size = 512
    self.fc1 = nn.Linear(input_size, hidden_size)
    self.fc2 = nn.Linear(hidden_size, hidden_size)
    self.fc3 = nn.Linear(hidden_size, num_classes)
    self.straight = nn.Linear(input_size, num_classes)
    self.softmax = nn.Softmax(dim=1)
    self.head_dropout = nn.Dropout(.2) # try 20% dropout 

  def forward(self, x):
    # print('x size before any gelu',x.size())
    x = torch.mean(x, dim=1, dtype=x.dtype) # do average pooling on patch-level reprs
    # x = F.gelu(self.fc1(x))
    # x = F.gelu(self.fc2(x)) 

    # add dropout
    x = self.head_dropout(x)

    # add layer norm
    x = F.layer_norm(x, (x.size(-1),)) # do not touch the BATCH SIZE dimension
                                        # but normalize over feature dim
    
    x = self.softmax(self.straight(x))
    return x

class Both(nn.Module):
  def __init__(self, encoder, num_classes):
    super(Both, self).__init__()
    self.encoder = encoder
    # Freeze encoder so that it is not trained
    for param in encoder.parameters():
      param.requires_grad = False # do ONLY linear probing
    self.head = ClassifierHead(EMBED_DIMS, num_classes)

  def forward(self, x):
    # with torch.no_grad(): 
    x = self.encoder(x)
    x = self.head(x)
    return x


model = Both(encoder, NUM_CLASSES)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)


# let's train it!!!
criterion = nn.CrossEntropyLoss()
optim = optim.AdamW(model.parameters(), lr=LR)

# Define transformations to be applied to the images
transform = transforms.Compose([
  transforms.Resize((IMG_CROPSIZE, IMG_CROPSIZE)), # Resize images to the same size
  transforms.ToTensor(), # Convert images to PyTorch tensors
  transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5]) # Normalize the images
])

train_dataset = ImageFolder(root=train_data_path, transform=transform)
val_dataset = ImageFolder(root=val_data_path, transform=transform)


# Uncomment the following lines if you want to load a subset of the dataset
# for faster data processing, but worse accuracy ofc
#--
# subset_size = 64
# total_size_tr = len(train_dataset)
# total_size_val = len(val_dataset)
# subset_dataset_tr, _ = random_split(train_dataset, [subset_size, total_size_tr-subset_size])
# subset_dataset_val, _ = random_split(val_dataset, [subset_size, total_size_val-subset_size])
# train_loader = DataLoader(subset_dataset_tr, batch_size=batch_size, shuffle=True)
# val_loader = DataLoader(subset_dataset_val, batch_size=batch_size)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)


def save_checkpoint(model, optim, epoch, save_path, checkpoint_freq=50):
  '''Save a checkpoint of a given model & an optimizer. 
  Every `checkpoint_freq` epochs save the model in a different file as well for post-use'''
  save_dict = {
      'model': model.state_dict(),
      'opt': optim.state_dict(),
      'epoch': epoch,
  }
  ep = epoch + 1 # temp epoch to avoid alchemy with string formats :)
  torch.save(save_dict, save_path+'-latest.pth.tar')
  save_path = save_path + f'-ep{ep}.pth.tar'
  if (ep) % checkpoint_freq == 0:
      torch.save(save_dict, save_path)

start_time = time.perf_counter() # 
for epoch in range(NUM_EPOCHS):
  epoch_start_time = time.perf_counter()
  model.train() # set the model to training mode
  running_loss = 0.0
  train_correct = 0
  total_train = 0
  for inputs, labels in train_loader:
    # send data to appropriate device
    inputs, labels = inputs.to(device), labels.to(device)

    optim.zero_grad() # set grads to zero
    outputs = model(inputs) # predictions
    
    _, predicted = outputs.max(dim=1) # we do not care about the values (underscore)
    
    train_correct += (predicted == labels).sum().item()
    total_train += labels.size(0)

    loss = criterion(outputs, labels) # compute the loss
    loss.backward() # backward pass
    optim.step() # update weights
    running_loss += loss.item()

  train_accuracy = train_correct / total_train

  # calculate average loss for the epoch
  epoch_loss = running_loss / total_train

  # Validation
  model.eval() # set the model to eval mode
  val_correct = 0
  total_val = 0
  with torch.no_grad():
    for inputs, labels in val_loader:
      inputs, labels = inputs.to(device), labels.to(device) # move data to device
     
      outputs = model(inputs)
      _, predicted = outputs.max(dim=1) # underscore is the values
      total_val += labels.size(0)
      val_correct += (predicted == labels).sum().item()
  
  time_taken = time.perf_counter() - epoch_start_time
  duration = timedelta(seconds=time_taken)
  val_accuracy = val_correct / total_val


  # print('val_correct',val_accuracy)
  # print('len(val_dataset)',len(val_dataset))

  print('Epoch: %d/%d' % (epoch+1, NUM_EPOCHS),
        'Train accuracy: %e' % train_accuracy,
        # 'Train correct: %d' % train_correct,
        # 'Train total: %d' % len(train_dataset),
        'Validation accuracy: %e' % val_accuracy, 
        # 'Val correct: %d' % val_correct,
        # 'Val total: %d' % len(val_dataset),
        'Loss %e' % epoch_loss,
        'Time taken:', duration)

  # print(f'Epoch {epoch+1}/{NUM_EPOCHS}, \nLoss: {epoch_loss}, \
  #       \nValidation accuracy: {val_accuracy}')
  # save model to disk 
  save_checkpoint(model, optim, epoch, SAVE_PATH, checkpoint_freq=100)

end_time = time.perf_counter()
total_duration=timedelta(seconds=end_time-start_time)
print('Total time taken', total_duration)
print('Done')

