import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from dataset import Braille_Dataset
from sampler import BalancedBatchSampler
from model import BrailleModel
from torchvision import transforms
from torch.utils.data import ConcatDataset

with_cuda = False
debug = False

hparams = {'batch_size': 16,
           'lr': 1e-5,
           'epochs': 50,
           'model_type': 'depth'}

if debug:
    hparams['batch_size'] = 32
    hparams['epochs'] = 100

class BrailleClassifier(pl.LightningModule):
    def __init__(self, dataset_path, hparams=None, model=None):
        super(BrailleClassifier, self).__init__()

        self.dataset_path = dataset_path
        self.model = model
        self.criterion = torch.nn.CrossEntropyLoss()
        self.batch_size = hparams['batch_size']

    def forward(self, x):
        if with_cuda:
            x = x.to(device='cuda')
        output = self.model(x)

    def training_step(self, batch, batch_idx):
        rgb = batch['rgb']
        depth = batch['depth']
        ytrue = batch['label']

        # shuffle batch
        idx = torch.randperm(ytrue.shape[0])
        rgb = rgb[idx].view(rgb.size())
        depth = depth[idx].view(depth.size())
        ytrue = ytrue[idx].view(ytrue.size())
        
        rgb = torch.transpose(rgb, 2, 3)
        depth = torch.transpose(depth, 2, 3)
        if with_cuda and torch.cuda.is_available():
            rgb = rgb.type(torch.cuda.FloatTensor)
            depth = depth.type(torch.cuda.FloatTensor)
            ytrue = ytrue.type(torch.cuda.LongTensor)
        else:
            rgb = rgb.float()
            depth = depth.float()
            ytrue = ytrue.long()
        
        if hparams['model_type']=="rgb":
            y_pred = self.model(rgb)
        else:
            y_pred = self.model(depth)

        loss= self.criterion(y_pred, ytrue.squeeze())
        _, ypred = torch.max(y_pred, 1)
        acc_train = sum(ytrue.squeeze() == ypred.squeeze()).item()
        acc_train = acc_train/len(ytrue)


        self.logger.experiment.add_scalar("Loss/Train", loss, self.global_step)
        self.logger.experiment.add_scalar("Accuracy/Train", acc_train, self.global_step)

        tensorboard_logs = {'train_mse_loss': loss,
                            'accuracy_train': acc_train}

        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        rgb = batch['rgb']
        depth = batch['depth']
        ytrue = batch['label']

        # shuffle batch
        idx = torch.randperm(ytrue.shape[0])
        rgb = rgb[idx].view(rgb.size())
        depth = depth[idx].view(depth.size())
        ytrue = ytrue[idx].view(ytrue.size())
        
        rgb = torch.transpose(rgb, 2, 3)
        depth = torch.transpose(depth, 2, 3)
        if with_cuda and torch.cuda.is_available():
            rgb = rgb.type(torch.cuda.FloatTensor)
            depth = depth.type(torch.cuda.FloatTensor)
            ytrue = ytrue.type(torch.cuda.LongTensor)
        else:
            rgb = rgb.float()
            depth = depth.float()
            ytrue = ytrue.long()
        
        if hparams['model_type']=="rgb":
            y_pred = self.model(rgb)
        else:
            y_pred = self.model(depth)

        loss= self.criterion(y_pred, ytrue.squeeze())
        _, ypred = torch.max(y_pred, 1)
        acc_val = sum(ytrue.squeeze() == ypred.squeeze()).item()
        acc_val = acc_val/len(ytrue)


        self.logger.experiment.add_scalar("Loss/Val", loss, self.global_step)
        self.logger.experiment.add_scalar("Accuracy/Val", acc_val, self.global_step)

        tensorboard_logs = {'train_mse_loss': loss,
                            'accuracy_val': acc_val}

        return {'loss': loss, 'log': tensorboard_logs}


    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=hparams['lr'])

    def train_dataloader(self):
        aug_param =transforms.Compose([transforms.ToTensor(),transforms.Normalize(0.5,0.5)])
        # train_data = Braille_Dataset(path_data=self.dataset_path, resize=True, mode='train', transformer=None, debug=debug)
        train_data_aug = Braille_Dataset(path_data=self.dataset_path, resize=True, mode='train', transformer=aug_param, debug=debug)
        # train_data.rgb_dataset = train_data.rgb_dataset + train_data_aug.rgb_dataset
        # train_data.depth_dataset = train_data.depth_dataset + train_data_aug.depth_dataset
        # train_data.target_dataset = train_data.target_dataset + train_data_aug.target_dataset
        # concat_dataset = ConcatDataset((train_data, train_data_aug))

        train_loader = torch.utils.data.DataLoader(
                                dataset=train_data_aug,
                                batch_size=self.batch_size,
                                sampler=BalancedBatchSampler(train_data_aug.rgb_dataset, train_data_aug.target_dataset),
                                # shuffle=True,
                                num_workers=8)
        return train_loader

    def val_dataloader(self):
        aug_param =transforms.Compose([transforms.ToTensor(),transforms.Normalize(0.5,0.5)])
        val_data = Braille_Dataset(path_data=self.dataset_path, resize=True, mode='val', transformer=aug_param, debug=debug)

        val_loader = torch.utils.data.DataLoader(
                                dataset=val_data,
                                batch_size=self.batch_size,
                                sampler=BalancedBatchSampler(val_data.rgb_dataset, val_data.target_dataset),
                                # shuffle=False,
                                num_workers=8)
        return val_loader


def run_trainer():
    global hparams
    if not debug:
        path_dataset = "/home/chiguera/Dropbox/sync/UW/Courses/Computer Vision/Project/dataset/"
        braille_model = BrailleModel(fc_units=[4096, 1024, 25])
    else:
        # path_dataset = "/Users/carolina/Dropbox/sync/UW/Courses/Computer Vision/Project/data/"
        path_dataset = "/home/chiguera/Dropbox/sync/UW/Courses/Computer Vision/Project/data/"
        braille_model = BrailleModel(fc_units=[4096, 1024, 6])
    model = BrailleClassifier(dataset_path=path_dataset, hparams=hparams, model=braille_model)
    logger = TensorBoardLogger('tb_logs2', name='braille_model_depth_aug')
    if not with_cuda:
        trainer = Trainer(max_epochs=hparams['epochs'],
                        accelerator="cpu",
                        gpus=0,
                        logger=logger,
                        val_check_interval=1.0
                        )
    else:
        trainer = Trainer(max_epochs=hparams['epochs'],
                        accelerator="gpu",
                        gpus=1,
                        logger=logger,
                        val_check_interval=1.0
                        )
    trainer.fit(model)
    print("Saving  model ... ")
    trainer.save_checkpoint(f"./depth/norm/braille_model_{hparams['model_type']}.ckpt")
    print("Saved")

if __name__ == '__main__':
    run_trainer()