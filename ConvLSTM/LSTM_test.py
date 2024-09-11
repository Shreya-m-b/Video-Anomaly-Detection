import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from Seq2Seq import Seq2Seq
import cv2
from torch.utils.data import DataLoader
from PIL import Image
import os
from video_dataset2 import VideoDatasetWithFlows
import time
from sklearn.metrics import roc_auc_score
# Use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

test_videos = []

test_video_paths = ["C:/Users/shreya/OneDrive/Desktop/avenue/testing_videos/{:02d}.avi".format(i) for i in range(1, 22)]

model = Seq2Seq(num_channels=1, num_kernels=64, kernel_size=(3, 3), padding=(1, 1), activation="relu", frame_size=(64, 64), num_layers=3).to(device)

optim = Adam(model.parameters(), lr=0)

criterion = nn.MSELoss()

checkpoint = torch.load("checkpoint_new_100.pth")
# model.load_state_dict(checkpoint)
model.load_state_dict(checkpoint["model_state"])

model.eval()

threshold = 32

root = 'C:/Users/shreya/OneDrive/Desktop/avenue/Attribute_based_VAD/data/'
test_dataset = VideoDatasetWithFlows(dataset_name = 'avenue', root = root, train = False, normalize = False)


while threshold <= 32:
    since = time.time()
    result = []
    for video_path in test_video_paths:
        arr = []
        cap = cv2.VideoCapture(video_path)
        count = 0

        # print(video_path)
        
        while True:
            count = count + 1
            ret, frame = cap.read()
            if not ret:
                break
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            resized_frame = cv2.resize(gray_frame, (64, 64))
            arr.append(resized_frame)

            if count > 10:
                target = arr[count - 1]
                target = torch.tensor(target)
                target = target.view(1, 1, 64, 64)
                target = target.to(device)

                inp = arr[count - 11 : count - 1]
                inp = torch.tensor(np.array(inp))
                inp = inp.view(1, 1, 10, 64, 64)
                inp = inp / 255.0
                inp = inp.to(device)
                # print(inp.shape)
                output = model(inp)

                output = output * 255

                loss = criterion(output, target)
                if count==11:
                    for i in range(11):
                        result.append(loss.detach().cpu().numpy())    
                else: 
                    result.append(loss.detach().cpu().numpy())

        cap.release()
        test_videos.append(np.array(arr))
    
    threshold = threshold + 1

    result=np.array(result)
    print(f'len of result={len(result)} and num of frames={test_dataset.num_of_frames}')
    for i in range(15):
        print(result[i])
    print(result)
    np.save('conv_lstm.npy',result)