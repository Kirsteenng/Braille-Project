import numpy as np
import cv2

if __name__ == "__main__":
    letter = 'A'
    path = f'{letter}_g2.npz'
    data = np.load(path, allow_pickle=True)

    rgb_img = data['img']
    depth_img= data['depth']

    num_img_rgb, h_rgb, w_rgb, _ = rgb_img.shape
    num_img_d, h_d, w_d, _ = depth_img.shape

    num_img = num_img_rgb if num_img_rgb <= num_img_d else num_img_d

    for i in range(num_img):
        img = cv2.hconcat([rgb_img[i], depth_img[i]])
        cv2.imshow('Gelslim', img)
        cv2.waitKey(5)