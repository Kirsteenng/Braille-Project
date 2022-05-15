import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from gen_dataset import Braille_Dataset
from sampler import BalancedBatchSampler
from model import CNN_Network

if __name__ == '__main__':
    BATCH_SIZE = 32
    FC_UNITS = [9216, 128, 6]
    lr = 1e-4
    max_epoch = 10
    model_name = "model_only_depth"

    dataset = Braille_Dataset(path_data='./data/', resize=True)
    train_dataloader = torch.utils.data.DataLoader(
                            dataset,
                            sampler=BalancedBatchSampler(dataset.depth_dataset, dataset.target_dataset),
                            batch_size=BATCH_SIZE)
    
    model = CNN_Network(fc_units=FC_UNITS)
    print(repr(model))

    if torch.cuda.is_available():
        model = model.cuda()

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    loss_list = []
    acc_list = []

    for epoch in range(0, max_epoch):
        model.train()
        loss_train = 0
        n_right_train = 0

        for step, sample_batched in enumerate(train_dataloader):
            # rgb = sample_batched['rgb'] 
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

            model.zero_grad()
            y_pred = model(depth)

            loss= loss_fn(y_pred, ytrue.squeeze())
            loss.backward()
            optimizer.step()
            
            loss_train += loss.item() * BATCH_SIZE
            _, ypred = torch.max(y_pred, 1)
            n_right_train += sum(ytrue.squeeze() == ypred.squeeze()).item()

            freq = 20
            if (step+1) % freq == 0:
                loss_train_reduced = loss_train / (freq*BATCH_SIZE)
                train_accuracy = float(n_right_train) / (freq*BATCH_SIZE)
                loss_list.append(loss_train_reduced)
                acc_list.append(train_accuracy)
                loss_train = 0
                n_right_train = 0
                print ('==================================================================')
                print ('[TRAIN set] Epoch {}, Step {}, Loss: {:.6f}, , Acc: {:.4f}'
                       .format(epoch, step + 1, loss_train_reduced, train_accuracy))
            
    plt.figure()
    plt.subplot(211)
    plt.plot(loss_list)
    plt.title('Loss: Cross Entropy')
    plt.ylabel('Loss')
    plt.subplot(212)
    plt.plot(acc_list)
    plt.title('Accuracy 6 classes')
    plt.ylabel('Accuracy')
    plt.xlabel('Iterations')
    plt.show()