import numpy as np
import cv2
import random
import torch
from torch.utils.data import Dataset

letters = ['A', 'B', 'C', 'D', 'E', 'F']
labels =   [0,   1,   2,   3,   4,   5 ]

class Braille_Dataset(Dataset):
    def __init__(self, path_data='./data', gray_depth=False, resize=False):
        self.rgb_dataset = []
        self.depth_dataset = []
        self.target_dataset= []
        self.w = 367
        self.h = 260
        print('Generating dataset')
        for l in range(len(letters)):
            path = f'{path_data}/{letters[l]}_g2.npz'
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
            self.rgb_dataset = self.rgb_dataset + list(rgb_imgs[0:num_img])
            self.depth_dataset = self.depth_dataset + list(depth_imgs[0:num_img])
            self.target_dataset = self.target_dataset + [np.array([labels[l]])]*num_img
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

        sample = {'rgb':torch.from_numpy(rgb), 
                  'depth': torch.from_numpy(depth), 
                  'label': torch.from_numpy(target)}

        return sample

if __name__ == '__main__':
    dataset = Braille_Dataset(gray_depth=False, resize=True)
    print(f'Length dataset: {dataset.__len__()}')
    sample = dataset.__getitem__(i=0)
    letter = letters[sample['label']]
    print(f'Sample for letter {letter}')
    cv2.imshow(f'RGB letter {letter}', sample['rgb'].numpy())
    cv2.imshow(f'Depth letter {letter}', sample['depth'].numpy())
    cv2.waitKey()