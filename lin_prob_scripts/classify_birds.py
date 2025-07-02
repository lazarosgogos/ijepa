from src.models.vision_transformer import vit_huge
from src import helper
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import torch.nn.functional as F
# Initialize the ViT-H model with the specified patch size and resolution
model = vit_huge(patch_size=4, num_classes=1000) # Adjust num_classes if needed


encoder, predictor = helper.init_model(device='cuda', 
                                       patch_size=4,
                                       model_name='vit_huge',
                                       crop_size=64,
                                       pred_depth=12,
                                       pred_emb_dim=384)



# Load the state dictionary from the file
load_path = 'logs/birds/jepa-latest.pth.tar'
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
print('INFO Gogos - Printing the predictor\'s architecture.')
print_model_layers(predictor) # 
print('Done with predictor\'s architecture.')


class ClassifierHead(nn.Module):
  def __init__(self, input_size, num_classes):
    super(ClassifierHead, self).__init__()
    hidden_size = 512
    self.fc1 = nn.Linear(input_size, hidden_size)
    # self.fc2 = nn.Linear(hidden_size, hidden_size)
    self.fc3 = nn.Linear(hidden_size, num_classes)
    self.softmax = nn.Softmax(dim=1) # this might be problematic

  def forward(self, x):
    x = F.gelu(self.fc1(x))
    # x = F.gelu(self.fc2(x)) 
    x = self.softmax(self.fc3(x))
    return x

class Both(nn.Module):
  def __init__(self, encoder, num_classes):
    super(Both, self).__init__()
    self.encoder = encoder
    # Freeze encoder so that it is not trained
    for param in encoder.parameters():
      param.requires_grad = False
    self.head = ClassifierHead(1280, num_classes)

  def forward(self, x):
    # with torch.no_grad(): 
    x = self.encoder(x)
    x = self.head(x)
    return x

num_classes = 525
model = Both(encoder, num_classes)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

lr = 0.01
num_epochs = 200
batch_size = 32
CROP_SIZE = 224

# let's train it!!!
criterion = nn.CrossEntropyLoss()
optim = optim.Adam(model.parameters(), lr=lr)

# Define transformations to be applied to the images
transform = transforms.Compose([
  transforms.Resize((CROP_SIZE, CROP_SIZE)), # Resize images to the same size
  transforms.ToTensor(), # Convert images to PyTorch tensors
  transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5]) # Normalize the images
])

# Define paths to datasets
train_data_path = 'datasets/birds/train'
val_data_path = 'datasets/birds/val'

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

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)


def save_checkpoint(model, optim, epoch, save_path, checkpoint_freq=50):
  '''Save a checkpoint of a given model & an optimizer. 
  Every 50 epochs save the model in a different file as well for post-use'''
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

for epoch in range(num_epochs):
  model.train() # set the model to training mode
  running_loss = 0.0
  for inputs, labels in train_loader:
    # send data to appropriate device
    inputs, labels = inputs.to(device), labels.to(device)

    # perform one-hot-encoding
    one_hot_labels = F.one_hot(labels, num_classes=num_classes).to(device)
    
    optim.zero_grad() # set grads to zero
    outputs = model(inputs) # predictions
    
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
    
      one_hot_labels = F.one_hot(labels, num_classes=num_classes).to(device)
      
      outputs = model(inputs)
      _, predicted = torch.max(outputs, 1)
      val_correct += (predicted == one_hot_labels).sum().item()
  
  # print('val_correct',val_correct)
  # print('len(val_dataset)',len(val_dataset))

  val_accuracy = val_correct / len(val_dataset)
  print(f'Epoch {epoch+1}/{num_epochs}, \nLoss: {epoch_loss}, \
        \nValidation accuracy: {val_accuracy}')
  # save model to disk 
  save_path = 'jepa_birds_classifier'
  save_checkpoint(model, optim, epoch, save_path)

print('Done')

