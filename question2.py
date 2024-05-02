# -*- coding: utf-8 -*-
"""
# Assignment 5

This is an basecode for assignment 5 of Artificial Intelligence class (CSCE-4613), Spring 2024
"""

import torch
import torchvision
import torch.nn as nn
from torchvision import transforms

from PIL import Image
import pickle
import matplotlib.pyplot as plt

"""## Question 2

### Define Training Data Loader
"""

train_batch_size = 32
train_dataset = torchvision.datasets.CIFAR10(root = "data/CIFAR-10",
                                             train = True,
                                             transform = transforms.ToTensor(),
                                             download = True)
train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size = train_batch_size,
                                           shuffle = True)

"""### Define Model and Training Framework"""

cuda = torch.cuda.is_available()
model = torchvision.models.resnet18(pretrained=False, num_classes = 10)
if cuda:
  model.cuda()

model.train()
learning_rate = 0.001
num_epochs = 1
optim = torch.optim.Adam(model.parameters(), lr = learning_rate)
loss_fn = nn.CrossEntropyLoss()
loss_logger = []
accuracy_logger = []

for epoch in range(1, num_epochs + 1):
  for it, (images, labels) in enumerate(train_loader):
      if cuda:
        images = images.cuda()
        labels = labels.cuda()

      outputs = model(images)
      predictions = torch.argmax(outputs, dim=1)
      accuracy = (predictions == labels).float().mean() * 100
      loss = loss_fn(outputs, labels)

      optim.zero_grad()
      loss.backward()
      optim.step()

      loss = loss.item()
      accuracy = accuracy.item()

      loss_logger.append(loss)
      accuracy_logger.append(accuracy)

      if it % 200 == 0:
        print("Epoch [%d/%d]. Iter [%d/%d]. Loss: %0.4f. Accuracy: %.2f" % (epoch, num_epochs, it + 1, len(train_loader), loss, accuracy))

torch.save(model.state_dict(), "CIFAR10-ResNet18.pth")

plt.figure()
plt.plot(loss_logger)
plt.title("Training Losses")
plt.show()

plt.figure()
plt.plot(accuracy_logger)
plt.title("Training Accuracies")
plt.show()

"""### Load Model and Evaluate Model On Testing Dataset"""

cuda = torch.cuda.is_available()
model = torchvision.models.resnet18(pretrained=False, num_classes = 10)
if cuda:
  model.cuda()

model.load_state_dict(torch.load("CIFAR10-ResNet18.pth"))
model.eval()

test_dataset = torchvision.datasets.CIFAR10(root = "data/CIFAR-10",
                                             train = False,
                                             transform = transforms.ToTensor(),
                                             download = True)
test_loader = torch.utils.data.DataLoader(test_dataset,
                                           batch_size = 1,
                                           shuffle = True)

final_accuracy = 0.0
for it, (images, labels) in enumerate(test_loader):
  if cuda:
    images = images.cuda()
    labels = labels.cuda()

  outputs = model(images)
  predictions = torch.argmax(outputs, dim=1)
  accuracy = (predictions == labels).float().mean() * 100

  accuracy = accuracy.item()
  final_accuracy += accuracy
  if it % 500 == 0:
    print("Iter [%d/%d]. Accuracy: %.2f" % (it + 1, len(test_loader), accuracy))

print("Final Accuracy: %0.2f" % (final_accuracy / len(test_loader)))

