from src.models.vision_transformer import vit_huge
from src import helper
# Initialize the ViT-H model with the specified patch size and resolution
model = vit_huge(patch_size=4, num_classes=1000) # Adjust num_classes if needed


encoder, predictor = helper.init_model(device='cuda', 
                                       patch_size=14,
                                       model_name='vit_huge',
                                       crop_size=224,
                                       pred_depth=12,
                                       pred_emb_dim=384)



import torch
# Load the state dictionary from the file
load_path = 'pretrained_models/vith.pth.tar'
ckpt = torch.load(load_path, map_location=torch.device('cpu'))
# state_dict = torch.load('/content/IN1K-vit.h.14-300e.pth.tar')
pretrained_dict = ckpt['encoder']

# encoder = vit_huge(patch_size=4)
# print(pretrained_dict)

# -- loading encoder
for k, v in pretrained_dict.items():
  encoder.state_dict()[k[len('module.'):]].copy_(v) 

import torch.nn as nn
import torch.nn.functional as F

# We already have an encoder loaded
# output neurons is 1280
# let's try adding a head

class ClassifierHead(nn.Module):
  def __init__(self, input_size, num_classes):
    super(ClassifierHead, self).__init__()
    hidden_size = 512
    self.fc1 = nn.Linear(input_size, hidden_size )
    self.fc2 = nn.Linear(hidden_size, num_classes)
    self.softmax = nn.Softmax(dim=1) # this might be problematic

  def forward(self, x):
    # x = torch.flatten(x, 1) # flatten the output
    x = self.fc1(x)
    x = F.gelu(x) # cause why not
    x = self.fc2(x)
    x = self.softmax(x)
    return x

class Both(nn.Module):
  def __init__(self, encoder, num_classes):
    super(Both, self).__init__()
    self.encoder = encoder
    self.head = ClassifierHead(1280, num_classes)

  def forward(self, x):
    x = self.encoder(x)
    x = self.head(x)
    return x

num_classes = 200

model = Both(encoder, num_classes)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
# let's train it!!!

import torch.optim as optim
from torch.utils.data import DataLoader, random_split

lr = 0.001
num_epochs = 1
batch_size = 32

criterion = nn.CrossEntropyLoss()
optim = optim.Adam(model.parameters(), lr=lr)

import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

# Define transformations to be applied to the images
transform = transforms.Compose([
  transforms.Resize((224, 224)), # Resize images to the same size
  transforms.ToTensor(), # Convert images to PyTorch tensors
  transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5]) # Normalize the images
])

# Define paths to datasets
train_data_path = 'datasets/tinyimagenet/train'
val_data_path = 'datasets/tinyimagenet/val'

train_dataset = ImageFolder(root=train_data_path, transform=transform)
val_dataset = ImageFolder(root=val_data_path, transform=transform)

# Uncomment the following lines if you want to load a subset of the dataset
# for faster data processing, but worse accuracy ofc

# subset_size = 64
# total_size_tr = len(train_dataset)
# total_size_val = len(val_dataset)
# subset_dataset_tr, _ = random_split(train_dataset, [subset_size, total_size_tr-subset_size])
# subset_dataset_val, _ = random_split(val_dataset, [subset_size, total_size_val-subset_size])
# train_loader = DataLoader(subset_dataset_tr, batch_size=batch_size, shuffle=True)
# val_loader = DataLoader(subset_dataset_val, batch_size=batch_size)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)


for epoch in range(num_epochs):
  model.train() # set the model to training mode
  running_loss = 0.0
  for inputs, labels in train_loader:
    inputs, labels = inputs.to(device), labels.to(device)
    # one_hot_labels = torch.nn.functional.one_hot(labels)
    one_hot_labels = F.one_hot(labels).to(device)
    # reshape if not all labels were found in this batch
    if (one_hot_labels.shape[1] < 200):
      diff = 200 - one_hot_labels.shape[1]
      zeros_tensor = torch.zeros(one_hot_labels.shape[0], diff).to(device)
      one_hot_labels = torch.cat((one_hot_labels, zeros_tensor), dim=1).type(torch.LongTensor).to(device)
    optim.zero_grad() # set grads to zero
    outputs = model(inputs)
    # print('outputs:', outputs)
    # print('one_hot_labels:', one_hot_labels)
    # break
    loss = criterion(outputs, one_hot_labels) # compute the loss
    loss.backward() # backward pass
    optim.step() # update weights
    running_loss += loss.item() * inputs.size(0)
  
  # calculate average loss for the epoch
  epoch_loss = running_loss / len(train_dataset)

  # Validation
  model.eval() # set the model to eval mode
  val_correct = 0
  with torch.no_grad():
    for inputs, labels in val_loader:
      inputs, labels = inputs.to(device), labels.to(device) # move data to device
    
      one_hot_labels = F.one_hot(labels).to(device)
      # reshape if not all labels were found in this batch
      if (one_hot_labels.shape[1] < 200):
        diff = 200 - one_hot_labels.shape[1]
        zeros_tensor = torch.zeros(one_hot_labels.shape[0], diff).to(device)
        one_hot_labels = torch.cat((one_hot_labels, zeros_tensor), dim=1).type(torch.LongTensor).to(device)
      outputs = model(inputs)
      _, predicted = torch.max(outputs, 1)
      val_correct += (predicted == one_hot_labels).sum().item()
  
  # print('val_correct',val_correct)
  # print('len(val_dataset)',len(val_dataset))

  val_accuracy = val_correct / len(val_dataset)
  print(f'Epoch {epoch+1}/{num_epochs}, \nLoss: {epoch_loss}, \
        \nValidation accuracy: {val_accuracy}')

print('Done')