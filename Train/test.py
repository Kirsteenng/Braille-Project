import numpy as np
import matplotlib.pyplot as plt
import torch
import time
import torch.nn as nn
from dataset2 import Braille_Dataset
from sampler import BalancedBatchSampler
from model import BrailleModel
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import seaborn as sn
import pandas as pd

FC_UNITS = [4096, 1024, 25]
BATCH_SIZE = 32
path_file = "./only_rgb/"
model_name = 'rgb'


if __name__ == '__main__':
    path = "/home/chiguera/Dropbox/sync/UW/Courses/Computer Vision/Project/dataset/"
    dataset = Braille_Dataset(path_data=path, resize=True, mode='test')
    test_dataloader = torch.utils.data.DataLoader(
                            dataset,
                            sampler=BalancedBatchSampler(dataset.depth_dataset, dataset.target_dataset),
                            batch_size=BATCH_SIZE)

    model = model = BrailleModel(fc_units=FC_UNITS)
    checkpoint_path = f"{path_file}/braille_model_{model_name}.ckpt"
    # checkpoint_path = f"./save_model/braille_model_depth.ckpt"
    checkpoint = torch.load(checkpoint_path)
    state_dict = {}
    for old_key in checkpoint['state_dict'].keys():
        new_key = old_key[len('model.'):]
        state_dict[new_key] = checkpoint['state_dict'][old_key]
    model.load_state_dict(state_dict)

    if torch.cuda.is_available():
        model = model.cuda()
        
    # evaluate the model on test dataset
    y_pred = []
    y_true = []

    # iterate over test data
    # model.eval()

    for param in model.parameters():
        param.grad = None

    n_right_train = 0
    with torch.no_grad():
        for batch in test_dataloader:
            rgb = batch['rgb']
            depth = batch['depth']
            ytrue = batch['label']

            # shuffle batch
            idx = torch.randperm(ytrue.shape[0])
            rgb = rgb[idx].view(rgb.size())
            depth = depth[idx].view(depth.size())
            ytrue = ytrue[idx].view(ytrue.size())
            
            rgb = torch.transpose(rgb, 1, 3)
            depth = torch.transpose(depth, 1, 3)
            if torch.cuda.is_available():
                rgb = rgb.type(torch.cuda.FloatTensor)
                depth = depth.type(torch.cuda.FloatTensor)
                ytrue = ytrue.type(torch.cuda.LongTensor)
            else:
                rgb = rgb.float()
                depth = depth.float()
                ytrue = ytrue.long()

            output = model(rgb) # Feed Network

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

    print(classification_report(y_true, y_pred, target_names=classes))

    print('[TEST] Generating confusion matrix')


    # Build confusion matrix
    cf_matrix = confusion_matrix(y_true, y_pred)
    # cf_matrix = cf_matrix/np.sum(cf_matrix)*100
    df_cm = pd.DataFrame(cf_matrix.astype(int), index = [i for i in classes],
                        columns = [i for i in classes])
    plt.figure(figsize = (22,18))
    sn.heatmap(df_cm, annot=True, fmt="d", cmap="YlGnBu")
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')
    # plt.show()
    plt.savefig(f'{path_file}/confusion_matrix_{model_name}_light.png')

    print('[TEST] Done')