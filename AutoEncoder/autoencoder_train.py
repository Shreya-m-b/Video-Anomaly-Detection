import torch
import torch.nn as nn
import os
from sklearn.neighbors import KernelDensity

import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import numpy as np
import random
import time


mean = np.array([0.5, 0.5, 0.5])
std = np.array([0.25, 0.25, 0.25])

data_transforms = {
    'testing': transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        transforms.Lambda(lambda x: torch.clamp(x, -1, 1))
    ]),
    'training': transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        transforms.Lambda(lambda x: torch.clamp(x, -1, 1))
    ]),
    'validation': transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        transforms.Lambda(lambda x: torch.clamp(x, -1, 1))
    ])
}


data_dir = 'images'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),data_transforms[x])for x in ['testing','training', 'validation']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=64,shuffle=True, num_workers=0)for x in ['testing','training', 'validation']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['testing','training', 'validation']}
class_names = image_datasets['training'].classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def imshow(inp, title):
    
    inp = inp.numpy().transpose((1, 2, 0))
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    plt.title(title)
    plt.show()


# Get a batch of training data
inputs, classes = next(iter(dataloaders['training']))
print(torch.min(inputs), torch.max(inputs))

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 4, padding = 1, stride = 2),
            nn.ReLU(),
            nn.Conv2d(16, 32, 4, padding = 1, stride = 2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, padding = 1, stride = 2),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, padding = 1, stride = 2),
            nn.ReLU(),
            nn.Conv2d(128, 256, 8)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 8),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, padding = 1, stride = 2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, padding = 1, stride = 2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 4, padding = 1, stride = 2),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, 4, padding = 1, stride = 2),
            nn.Tanh()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


model = Autoencoder().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)


num_epochs = 10
outputs = []
for epoch in range(num_epochs):
    since = time.time()
    for image, label in dataloaders["training"]:
        image = image.to(device)
        label = label.to(device)

        result = model(image)
        loss = criterion(result, image)

        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

    elapsed = time.time() - since
    print(f"epoch {epoch + 1}/{num_epochs}, loss = {loss.item()}, time taken = {elapsed:.4f}")
    outputs.append((epoch, image, result))

    if epoch + 1 == 5:
        checkpoint = {
            "epoch": epoch + 1,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict()
        }
        torch.save(checkpoint, "checkpoint.pth")


torch.save(model.state_dict(), "model.pth")


for i in range(num_epochs):
    plt.figure(figsize = (9, 2))
    # plt.gray()
    images = outputs[i][1].cpu().detach().numpy()
    reconstructed_images = outputs[i][2].cpu().detach().numpy()

    for num, item in enumerate(images):
        if num >= 9:
            break

        plt.subplot(2, 9, num + 1)
        newitem = np.transpose(item, (1, 2, 0))
        newitem = newitem / 2 + 0.5
        plt.imshow(newitem)

    for num, item in enumerate(reconstructed_images):
        if num >= 9:
            break

        plt.subplot(2, 9, 9 + num + 1)
        newitem = np.transpose(item, (1, 2, 0))
        newitem = newitem / 2 + 0.5
        plt.imshow(newitem)


plt.show()
