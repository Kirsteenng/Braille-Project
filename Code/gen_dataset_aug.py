import numpy as np
import cv2
import random
import torch
import sys
from torch.utils.data import Dataset

# retake letters S and T
<<<<<<< HEAD:Code/gen_dataset_aug.py
letters = ['A', 'B', 'C', 'D', 'E', 'F']
labels =   list(range(0, len(letters)))

class Braille_Dataset(Dataset):
    def __init__(self, path_data='/Users/Kirsteenng_1/Desktop/UW courses/MSDS/Spring 2022/CSE 576/Project/A_Z', gray_depth=False, resize=False, mode='train',transformer=None): #mode=['train', 'val', 'test']
        self.rgb_dataset = []
        self.depth_dataset = []
        self.target_dataset= []
        self.w = 183
        self.h = 130
        self.num_classes = len(labels)
        self.labels = letters
        self.transformer=transformer
=======
letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'U', 'V', 'W', 'X', 'Y', 'Z', "num"]
labels =   list(range(0, len(letters)))

class Braille_Dataset(Dataset):
    def __init__(self, path_data='./data', gray_depth=False, resize=False, mode='train'): #mode=['train', 'val', 'test']
        self.rgb_dataset = []
        self.depth_dataset = []
        self.target_dataset= []
        self.w = 367
        self.h = 260
        self.num_classes = len(labels)
        self.labels = letters
>>>>>>> caf2cc7c4ca884bab1e4ff397c889c229694fbee:Code/gen_dataset.py
        print(f'Generating dataset - Mode: {mode}')
        for l in range(len(letters)):
            path = f'{path_data}/{letters[l]}_g2.npz'
            print(f'Loading {path}')
            data = np.load(path, allow_pickle=True)
            rgb_imgs = data['img']
            depth_imgs= data['depth']
            num_img_rgb = rgb_imgs.shape[0]
            num_img_d = depth_imgs.shape[0]
            
            if resize:
                rgb_resize = np.zeros([num_img_rgb, self.h//2, self.w//2, 3])
                for n in range(num_img_rgb):
                    rgb_resize[n]=cv2.resize(rgb_imgs[n]/255., (self.w//2, self.h//2), interpolation=cv2.INTER_LINEAR)
                rgb_imgs = rgb_resize.copy()
                del rgb_resize
                depth_resize = np.zeros([num_img_d, self.h//2, self.w//2, 3])
                for n in range(num_img_d):
                    depth_resize[n]=cv2.resize(depth_imgs[n]/255., (self.w//2, self.h//2), interpolation=cv2.INTER_LINEAR)
                depth_imgs = depth_resize.copy()
                del depth_resize

            if gray_depth:
                depth_gray_imgs = np.zeros((depth_imgs.shape[0], depth_imgs.shape[1], depth_imgs.shape[2]))
                for n in range(num_img_d):
                    depth_gray_imgs[n] = cv2.cvtColor(depth_imgs[n], cv2.COLOR_BGR2GRAY)
                depth_imgs = depth_gray_imgs
            num_img = num_img_rgb if num_img_rgb <= num_img_d else num_img_d

            if(mode=='train'):
                num_img_train = int(num_img*0.7)
                self.rgb_dataset = self.rgb_dataset + list(rgb_imgs[0:num_img_train])
                self.depth_dataset = self.depth_dataset + list(depth_imgs[0:num_img_train])
                self.target_dataset = self.target_dataset + [np.array([labels[l]])]*num_img_train
            elif(mode=='val'):
                num_img_train = int(num_img*0.7)
                num_img_val = num_img_train + int(num_img*0.2)
                self.rgb_dataset = self.rgb_dataset + list(rgb_imgs[num_img_train:num_img_val])
                self.depth_dataset = self.depth_dataset + list(depth_imgs[num_img_train:num_img_val])
                self.target_dataset = self.target_dataset + [np.array([labels[l]])]*num_img_val
            elif(mode=='test'):
                num_img_train = int(num_img*0.7)
                num_img_test = num_img_train + int(num_img*0.2)
                self.rgb_dataset = self.rgb_dataset + list(rgb_imgs[num_img_test:])
                self.depth_dataset = self.depth_dataset + list(depth_imgs[num_img_test:])
                self.target_dataset = self.target_dataset + [np.array([labels[l]])]*(num_img-num_img_test)
            else:
                print("Mode not recognized")
                sys.exit(0)
        self.shuffle_dataset()
        print('')
        
    
    def shuffle_dataset(self):
        temp = list(zip(self.rgb_dataset, self.depth_dataset, self.target_dataset))
        random.shuffle(temp)
        res1, res2, res3 = zip(*temp)
        self.rgb_dataset, self.depth_dataset, self.target_dataset = list(res1), list(res2), list(res3)

    def __len__(self):
        return len(self.rgb_dataset)

    def __getitem__(self, i):
        rgb = self.rgb_dataset[i]
        depth = self.depth_dataset[i]
        target = self.target_dataset[i]
        
        if self.transformer is not None:
            rgb = self.transformer(torch.from_numpy(rgb))
            depth = self.transformer(torch.from_numpy(depth))
            
            sample = {'rgb':rgb, 
                      'depth': depth, 
                      'label': torch.from_numpy(target)}
            
        sample = {'rgb':torch.from_numpy(rgb), 
                  'depth': torch.from_numpy(depth), 
                  'label': torch.from_numpy(target)}
    
        return sample

if __name__ == '__main__':
<<<<<<< HEAD:Code/gen_dataset_aug.py
    dataset = Braille_Dataset(path_data="/Users/Kirsteenng_1/Desktop/UW courses/MSDS/Spring 2022/CSE 576/Project/A_Z", gray_depth=False, resize=True, mode='test')
=======
    dataset = Braille_Dataset(path_data="./dataset/", gray_depth=False, resize=True, mode='test')
>>>>>>> caf2cc7c4ca884bab1e4ff397c889c229694fbee:Code/gen_dataset.py
    print(f'Length dataset: {dataset.__len__()}')
    sample = dataset.__getitem__(i=0)
    letter = letters[sample['label']]
    print(f'Sample for letter {letter}')
    cv2.imshow(f'RGB letter {letter}', sample['rgb'].numpy())
    cv2.imshow(f'Depth letter {letter}', sample['depth'].numpy())
    cv2.waitKey()