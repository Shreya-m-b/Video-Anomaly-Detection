import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from Seq2Seq import Seq2Seq
import cv2
from torch.utils.data import DataLoader
import time

# Use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device='cpu'
print(device)

# Read AVI videos
train_videos = []

train_video_paths = ["C:/Users/shreya/OneDrive/Desktop/avenue/training_videos/{:02d}.avi".format(i) for i in range(1, 17)]

for video_path in train_video_paths:
    video = []
    cap = cv2.VideoCapture(video_path)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized_frame = cv2.resize(gray_frame, (64, 64))
        video.append(resized_frame)
    cap.release()
    train_videos.append(np.array(video))

# Convert to numpy array
train_videos = np.array(train_videos)

# Shuffle Data
np.random.shuffle(train_videos)

# Train, Test splits
train_data = train_videos[:16]

def collate(batch):
    # Sort the sequences in the batch by length (number of frames)
    batch = sorted(batch, key=lambda x: len(x), reverse=True)
    # Determine the maximum sequence length in the batch
    max_length = len(batch[0])

    # Pad sequences with the maximum length in the batch
    padded_batch = []
    for seq in batch:
        padding_frames = np.zeros((max_length - len(seq), *seq[0].shape), dtype=seq.dtype)
        padded_seq = np.concatenate([seq, padding_frames])
        padded_batch.append(padded_seq)

    # Convert to tensor, add channel dim, scale pixels between 0 and 1, send to GPU
    padded_batch = torch.tensor(padded_batch).unsqueeze(1)
    padded_batch = padded_batch / 255.0
    padded_batch = padded_batch.to(device)

    # Randomly pick 10 frames as input, 11th frame is target
    rand = np.random.randint(10, max_length)
    return padded_batch[:, :, rand - 10:rand], padded_batch[:, :, rand]

# Training Data Loader
train_loader = DataLoader(train_data, shuffle=True, batch_size=20, collate_fn=collate)

# The input video frames are grayscale, thus single channel
model = Seq2Seq(num_channels=1, num_kernels=64, kernel_size=(3, 3), padding=(1, 1),
                activation="relu", frame_size=(64, 64), num_layers=3).to(device)

optim = Adam(model.parameters(), lr=1e-4)

# Mean Squared Error Loss
criterion = nn.MSELoss()

num_epochs = 100

since=time.time()
for epoch in range(1, num_epochs + 1):
    train_loss = 0
    model.train()
    for batch_num, (input, target) in enumerate(train_loader, 1):
        input = input.to(device)
        target = target.to(device)
        
        output = model(input)
        loss = criterion(output, target)
        
        optim.zero_grad()
        loss.backward()
        optim.step()
        
        train_loss += loss.item()
    
    train_loss /= len(train_loader.dataset)
    elapsed=time.time()-since
    print("Epoch: {} Training Loss: {:.4f} time: {}".format(epoch, train_loss,elapsed))
