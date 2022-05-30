from cProfile import label
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
import pickle
from torchvision import transforms



def validation():
    global val_dataloader, model, loss_val_list, acc_val_list
    loss_val = 0
    n_right_val = 0

    for step, sample_batched in enumerate(val_dataloader):
        rgb = sample_batched['rgb']
        depth = sample_batched['depth']
        ytrue = sample_batched['label']

        # shuffle batch
        idx = torch.randperm(ytrue.shape[0])
        # rgb = rgb[idx].view(rgb.size())
        depth = depth[idx].view(depth.size())
        ytrue = ytrue[idx].view(ytrue.size())

        # rgb = torch.transpose(rgb, 1, 3)
        depth = torch.transpose(depth, 1, 3)
        if torch.cuda.is_available():
            # rgb = rgb.type(torch.cuda.FloatTensor)
            depth = depth.type(torch.cuda.FloatTensor)
            ytrue = ytrue.type(torch.cuda.LongTensor)
        model.eval()
        y_pred = model(depth)
        loss= loss_fn(y_pred, ytrue.squeeze())
        loss_val += loss.item() * BATCH_SIZE
        _, ypred = torch.max(y_pred, 1)
        n_right_val += sum(ytrue.squeeze() == ypred.squeeze()).item()

        if (step+1) % freq_stats_val == 0:
            loss_val_reduced = loss_val / (freq_stats_val*BATCH_SIZE)
            val_accuracy = float(n_right_val) / (freq_stats_val*BATCH_SIZE)
            loss_val_list.append(loss_val_reduced)
            acc_val_list.append(val_accuracy)
            loss_val = 0
            n_right_val = 0
            print ('==================================================================')
            print ('[VAL] Epoch {}, Step {}, Loss: {:.6f}, , Acc: {:.4f}'
                    .format(epoch, step + 1, loss_val_reduced, val_accuracy))

def training():
    torch.cuda.empty_cache()
    accum_iter = 20  
    global train_dataloader, model, loss_train_list, acc_train_list
    loss_train = 0
    n_right_train = 0

    for step, sample_batched in enumerate(train_dataloader):
        rgb = sample_batched['rgb'] 
        depth = sample_batched['depth']
        ytrue = sample_batched['label']

        # shuffle batch
        idx = torch.randperm(ytrue.shape[0])
        # rgb = rgb[idx].view(rgb.size())
        depth = depth[idx].view(depth.size())
        ytrue = ytrue[idx].view(ytrue.size())
        
        # rgb = torch.transpose(rgb, 1, 3)
        depth = torch.transpose(depth, 1, 3)
        if torch.cuda.is_available():
            # rgb = rgb.type(torch.cuda.FloatTensor)
            depth = depth.type(torch.cuda.FloatTensor)
            ytrue = ytrue.type(torch.cuda.LongTensor)

        model.train()
        model.zero_grad()
        y_pred = model(depth)

        loss= loss_fn(y_pred, ytrue.squeeze())
        #loss = loss / accum_iter 
        loss.backward()
        optimizer.step()
       
        # if ((step + 1) % accum_iter == 0) or (step + 1 == len(train_dataloader)):
        #     optimizer.step()
        #     model.zero_grad()
       
        loss_train += loss.item() * BATCH_SIZE
        _, ypred = torch.max(y_pred, 1)
        n_right_train += sum(ytrue.squeeze() == ypred.squeeze()).item()

        
        if (step+1) % freq_stats_train == 0:
            loss_train_reduced = loss_train / (freq_stats_train*BATCH_SIZE)
            train_accuracy = float(n_right_train) / (freq_stats_train*BATCH_SIZE)
            loss_train_list.append(loss_train_reduced)
            acc_train_list.append(train_accuracy)
            loss_train = 0
            n_right_train = 0
            print ('==================================================================')
            print ('[TRAIN] Epoch {}, Step {}, Loss: {:.6f}, , Acc: {:.4f}'
                    .format(epoch, step + 1, loss_train_reduced, train_accuracy))

##############################

if __name__ == '__main__':
    
    # hyperparameters
    BATCH_SIZE = 4
    FC_UNITS = [4096, 1024, 25]
    lr = 1e-5
    max_epoch = 50

    freq_stats_train = 20
    freq_stats_val = 20

    model_name = "model_only_depth"

# create train and val dataloaders
   # with open('/homes/iws/kirstng/CSE576/Project/Braille-Project/gen_data_A-Z_train.pickle','rb') as handle: #me
    #    train_dataset = pickle.load(handle) #me
   # with open('/homes/iws/kirstng/CSE576/Project/Braille-Project/gen_data_A-Z_val.pickle','rb') as handle: #me
   #     val_dataset = pickle.load(handle) #me
 #   with open('/homes/iws/kirstng/Braille-Project/gen_data_A-Z_test.pickle','rb') as handle: #me
 #       test_dataset = pickle.load(handle) #me
    aug_param =transforms.Compose([transforms.ToTensor(),transforms.Normalize(0.5,0.5)])
    train_dataset_1 = Braille_Dataset(path_data='/Users/Kirsteenng_1/Desktop/UW courses/MSDS/Spring 2022/CSE 576/Project/A_Z', resize=True, mode='train',transformer=None)
    train_dataset_2= Braille_Dataset(path_data='/Users/Kirsteenng_1/Desktop/UW courses/MSDS/Spring 2022/CSE 576/Project/A_Z', resize=True, mode='train',transformer=aug_param)
    train_dataset_rgb = Braille_Dataset(path_data='/Users/Kirsteenng_1/Desktop/UW courses/MSDS/Spring 2022/CSE 576/Project/A_Z', resize=True, mode='train',transformer=None)
    
    train_dataset_rgb.rgb_dataset = train_dataset_1.rgb_dataset + train_dataset_2.rgb_dataset
    train_dataset_rgb.depth_dataset = train_dataset_1.depth_dataset + train_dataset_2.depth_dataset
    train_dataset_rgb.target_dataset = train_dataset_1.target_dataset + train_dataset_2.target_dataset
    
   # 
    train_dataloader = torch.utils.data.DataLoader(
                            train_dataset_rgb,
                            sampler=BalancedBatchSampler(train_dataset_rgb.depth_dataset, train_dataset_rgb.target_dataset),
                            batch_size=BATCH_SIZE)

    val_dataset = Braille_Dataset(path_data='/Users/Kirsteenng_1/Desktop/UW courses/MSDS/Spring 2022/CSE 576/Project/A_Z', resize=True, mode='val')
   
    '''
    val_dataloader = torch.utils.data.DataLoader(
                            val_dataset,
                            sampler=BalancedBatchSampler(val_dataset.depth_dataset, val_dataset.target_dataset),
                            batch_size=BATCH_SIZE//2)
    
    # create model and show specs
    model = CNN_Network(fc_units=FC_UNITS)
    print(repr(model))

    # send model to gpu, if available
    if torch.cuda.is_available():
        print("Using CUDA")
        model = model.cuda()

    # we use cross entropy loss, given that we are working with a muticlass classification problem
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # some list to save losses and accuracy results
    loss_train_list = []
    acc_train_list = []
    loss_val_list = []
    acc_val_list = []

    # start timer to measure how long does the training takes
    time_start = time.time()

    # --- START OF TRAINING LOOP ---
    for epoch in range(0, max_epoch):
        print('Epoch #: ', epoch)
        training()
        validation()
    # --- END OF TRAINING LOOP ---

    # stop timer and measure enlapse time
    time_end = time.time()
    enlapse_time = (time_end - time_start) / 3600.
    print(f"Enlapse time training: {enlapse_time} hours")

'''
    # plot loss and accuracy in training and val data
    plt.figure()
    plt.subplot(211)
    #plt.plot(loss_train_list, label='Train')
    #plt.plot(loss_val_list, label='Val')
    plt.title('Loss: Cross Entropy')
    plt.ylabel('Loss')
    plt.xlabel('Iterations')
    plt.legend()
    '''
    plt.subplot(212)
    plt.plot(acc_train_list, label='Train')
    plt.plot(acc_val_list, label='Val')
    plt.title('Classification Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Iterations')
    plt.legend()
    plt.savefig('learning_curve_all.jpg')
    '''
    # save model
   # torch.save(model.state_dict(), 'braille_model_all.pth')

   
    #TODO
    # ADD VALIDATION/ TEST DATASET
    # CONFUSION MATRIX
    # SAVE BEST MODEL
    # EARLY STOPPING