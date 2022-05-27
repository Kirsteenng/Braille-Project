import numpy as np
import matplotlib.pyplot as plt
import torch
import time
import torch.nn as nn
from gen_dataset import Braille_Dataset
from sampler import BalancedBatchSampler
from model import CNN_Network
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd

FC_UNITS = [9216, 128, 25]
BATCH_SIZE = 64


if __name__ == '__main__':

    dataset = Braille_Dataset(path_data='./dataset/', resize=True, mode='test')
    test_dataloader = torch.utils.data.DataLoader(
                            dataset,
                            sampler=BalancedBatchSampler(dataset.depth_dataset, dataset.target_dataset),
                            batch_size=BATCH_SIZE)

    model = model = CNN_Network(fc_units=FC_UNITS)
    model.load_state_dict(torch.load('braille_model_all.pth'))

    if torch.cuda.is_available():
        model = model.cuda()
        
    # evaluate the model on test dataset
    y_pred = []
    y_true = []

    # iterate over test data
    model.eval()
    n_right_train = 0
    for sample_batched in test_dataloader:
        depth = sample_batched['depth']
        ytrue = sample_batched['label']

        idx = torch.randperm(ytrue.shape[0])
        depth = depth[idx].view(depth.size())
        ytrue = ytrue[idx].view(ytrue.size())
        
        # rgb = torch.transpose(rgb, 1, 3)
        depth = torch.transpose(depth, 1, 3)
        if torch.cuda.is_available():
            # rgb = rgb.type(torch.cuda.FloatTensor)
            depth = depth.type(torch.cuda.FloatTensor)
            ytrue = ytrue.type(torch.cuda.LongTensor)

        output = model(depth) # Feed Network

        # output = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
        _, output = torch.max(output, 1)
        output = output.data.cpu().numpy()
        y_pred.extend(output) # Save Prediction
        labels = ytrue.squeeze().data.cpu().numpy()
        y_true.extend(labels) # Save Truth

    # constant for classes
    classes = dataset.labels

    n_right_train = sum(np.array(y_true)==np.array(y_pred))
    print('[TEST] Accuracy in test dataset:')
    print(f'\t {n_right_train} out of {len(y_true)} samples')
    print(f'\t {n_right_train/len(y_true)}')

    print('[TEST] Generating confusion matrix')


    # Build confusion matrix
    cf_matrix = confusion_matrix(y_true, y_pred)
    # cf_matrix = cf_matrix/np.sum(cf_matrix)*100
    df_cm = pd.DataFrame(cf_matrix.astype(int), index = [i for i in classes],
                        columns = [i for i in classes])
    plt.figure(figsize = (12,7))
    sn.heatmap(df_cm, annot=True)
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')
    # plt.show()
    plt.savefig('confusion_matrix_all.png')

    print('[TEST] Done')