# -*- coding: utf-8 -*-
"""
# Assignment 4

This is an basecode for assignment 5 of Artificial Intelligence class (CSCE-4613), Spring 2024
"""

import torch
import torchvision
import torch.nn as nn
from torchvision import transforms

from PIL import Image
import pickle
import matplotlib.pyplot as plt

"""## Question 1
### Define Input Transformation
"""

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
IMAGE_SIZE = 224
image_transforms = transforms.Compose([
                           transforms.Resize(IMAGE_SIZE),
                           transforms.CenterCrop(IMAGE_SIZE),
                           transforms.ToTensor(),
                           transforms.Normalize(mean = MEAN, std = STD)])

index2name = pickle.load(open("imagenet_class_names.pkl", "rb"))

"""### Define Model"""

model = torchvision.models.resnet50(pretrained=True)
softmax_layer = nn.Softmax(dim=1)
model.eval()

"""### Classify and Visualize Image"""

image_path = "dog.jpg"
original_image = Image.open(image_path).convert("RGB")
image = image_transforms(original_image)
image = image.unsqueeze(0)
output = softmax_layer(model(image))

prediction = torch.argmax(output, dim=1).item()
prob = output[0, prediction].item() * 100
predicted_name = index2name[prediction]

plt.imshow(original_image)
plt.title("Class: %s. Probabilty: %.2f" % (predicted_name, prob) + "%")
plt.axis("off")
plt.show()

"""### Get Top-K Predictions"""

K = 5
indices = torch.argsort(output, dim = 1, descending=True)
for i in range(0, K):
  prob = output[0, indices[0, i].item()].item() * 100
  predicted_name = index2name[indices[0, i].item()]
  print("%d-th. Class: %s. Probabilty: %0.2f" % (i + 1, predicted_name, prob) + "%")


