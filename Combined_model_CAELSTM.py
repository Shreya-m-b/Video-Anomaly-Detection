import numpy as np
from video_dataset2 import VideoDatasetWithFlows
import time
from sklearn.metrics import roc_auc_score


result_ae=np.load('conv_ae/conv_ae.npy')
# print(result_ae)

min_ae=np.min(result_ae)
max_ae=np.max(result_ae)

norm_ae=(result_ae-min_ae)/(max_ae-min_ae)

# print(f'norm={norm_ae}')


result_lstm=np.load('conv_lstm/conv_lstm.npy')
# print(result_lstm)

min_lstm=np.min(result_lstm)
max_lstm=np.max(result_lstm)

norm_lstm=(result_lstm-min_lstm)/(max_lstm-min_lstm)

# print(f'norm={norm_lstm}')

factor_lstm=1
factor_ae=0.8

combined_arr=(norm_lstm*factor_lstm)+(norm_ae*factor_ae)

root = 'C:/Users/shreya/OneDrive/Desktop/Attribute_based_VAD/data/'
test_dataset = VideoDatasetWithFlows(dataset_name = 'avenue', root = root, train = False, normalize = False)

i=0.01
while(i<0.1):
    final=[]
    for j in combined_arr:
        if (j>i):
            final.append(1)
        else:
            final.append(0)
    print('threshold ', i , 'Micro AUC: ', roc_auc_score(test_dataset.all_gt, final) * 100,'factor_lstm=',factor_lstm,'factor_ae=',factor_ae)
    i+=0.005