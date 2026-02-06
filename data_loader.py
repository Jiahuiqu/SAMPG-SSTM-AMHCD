import os
import torch
import numpy as np
import cv2
import random
from dataloader.get_datasets import get_dataset
from torch.utils.data import Dataset, DataLoader

class HSICD_Dataset(Dataset):
    def __init__(self, dataset_name='China', mode='train', patch_size=9, bx='0'):
        super().__init__()

        self.dataset_name = dataset_name
        self.mode = mode
        self.padding = int(patch_size / 2)
        self.bx = bx
        # 全标数据集 0为unchanged 1为changed 其中对于River数据集，255为changed，已经将其gt除以255
        if self.dataset_name == 'China' or self.dataset_name == 'River':
            self.img_t1, self.img_t2, self.img_gt = get_dataset(self.dataset_name, self.bx)
            # self.img_t1, self.img_t2 = self.data_preprocess(self.img_t1), self.data_preprocess(self.img_t2)
            self.h, self.w = self.img_gt.shape[0], self.img_gt.shape[1]
            self.img_t1_padding = cv2.copyMakeBorder(self.img_t1, self.padding, self.padding, self.padding,
                                                     self.padding,
                                                     cv2.BORDER_REFLECT)
            self.img_t2_padding = cv2.copyMakeBorder(self.img_t2, self.padding, self.padding, self.padding,
                                                     self.padding,
                                                     cv2.BORDER_REFLECT)
            self.random_points = self.read_points()

        # 未全标数据集 1为changed 2为unchanged 所以得进行调整，将gt中的1变为2,2变为1
        elif self.dataset_name == 'BayArea' or self.dataset_name == 'Barbara' or self.dataset_name == 'Hermiston' :
            self.img_t1, self.img_t2, self.img_gt = get_dataset(self.dataset_name, self.bx)
            # self.img_t1, self.img_t2 = self.data_preprocess(self.img_t1), self.data_preprocess(self.img_t2)
            img_gt_tmp = np.zeros_like(self.img_gt)
            img_gt_tmp[self.img_gt == 1.] = 2.
            self.img_gt[self.img_gt == 2.] = 1.
            self.img_gt[img_gt_tmp == 2.] = 2.

            self.h, self.w = self.img_gt.shape[0], self.img_gt.shape[1]
            self.img_t1_padding = cv2.copyMakeBorder(self.img_t1, self.padding, self.padding, self.padding,
                                                     self.padding,
                                                     cv2.BORDER_REFLECT)
            self.img_t2_padding = cv2.copyMakeBorder(self.img_t2, self.padding, self.padding, self.padding,
                                                     self.padding,
                                                     cv2.BORDER_REFLECT)
            self.random_points = self.read_points()

        else:
            raise ValueError(f"Invalid dataset name {dataset_name}. Expected one of: China, Hermiston, River, "
                             f"BayArea, Barbara.")

    def read_points(self):
        points = []

        if self.mode != 'test':
            if self.bx == '0':
                file_name = os.path.join(f"datasets/{self.dataset_name}", f"{self.mode}_wr_2%.txt")  #wr:well reg   bx:旋转角度    sj：随机变形
                # file_name = os.path.join(f"datasets/{self.dataset_name}", f"{self.mode}_reg_1%.txt")  #wr:well reg   bx:旋转角度    sj：随机变形
            elif self.bx == 'affine':
                file_name = os.path.join(f"datasets/{self.dataset_name}", f"{self.mode}_wr_reg_1%.txt")  # wr:well reg   bx:旋转角度    sj：随机变形
            elif self.bx == '1':
                file_name = os.path.join(f"datasets/{self.dataset_name}", f"{self.mode}_bx_1_2%.txt")  # wr:well reg   bx:旋转角度    sj：随机变形
            elif self.bx == '3':
                file_name = os.path.join(f"datasets/{self.dataset_name}", f"{self.mode}_bx_3_2%.txt")  # wr:well reg   bx:旋转角度    sj：随机变形
            elif self.bx == '5':
                file_name = os.path.join(f"datasets/{self.dataset_name}", f"{self.mode}_bx_5_2%.txt")  # wr:well reg   bx:旋转角度    sj：随机变形
            # elif self.bx == 'sj':
            elif self.bx == 'sj' or self.bx == 'sj_noreg':
                file_name = os.path.join(f"datasets/{self.dataset_name}", f"{self.mode}_sj_2%.txt")  # wr:well reg   bx:旋转角度    sj：随机变形
                print('sj::::')
            elif self.bx == '0_reg':
                file_name = os.path.join(f"datasets/{self.dataset_name}", f"{self.mode}_wr_reg_1%.txt")  # wr:well reg   bx:旋转角度    sj：随机变形
                print('reg::::')
            elif self.bx == 'reg':
                file_name = os.path.join(f"datasets/{self.dataset_name}", f"{self.mode}_reg_1%.txt")  # wr:well reg   bx:旋转角度    sj：随机变形

            elif self.bx == 'sj_nopz':
                file_name = os.path.join(f"datasets/{self.dataset_name}", f"{self.mode}_reg_1%.txt")  # wr:well reg   bx:旋转角度    sj：随机变形
            with open(file_name, 'r') as file:
                for line in file:
                    point = int(line.strip())
                    print(point)
                    points.append(point)

        else:
            all_num = self.h * self.w
            if self.dataset_name == 'China' or self.dataset_name == 'River':
                points = list(range(all_num))
            elif self.dataset_name == 'BayArea' or self.dataset_name == 'Barbara' or self.dataset_name == 'Liyukou' or self.dataset_name == 'Hermiston':
                whole_point = self.img_gt.reshape(1, all_num)
                points = np.nonzero(whole_point[0])[0].tolist()

        return points

    def data_preprocess(self, img):
        mean = img.mean(axis=(0, 1))
        std = img.std(axis=(0, 1))

        img_normalized = (img - mean) / std

        return img_normalized

    def __len__(self):
        return len(self.random_points)

    def __getitem__(self, index):
        original_i, original_j = divmod(self.random_points[index], self.w)  # 第几行，第几列
        assert 0 <= original_i < self.h, f"索引{original_i}超出图像高度{self.h}"
        assert 0 <= original_j < self.w, f"索引{original_j}超出图像宽度{self.w}"
        new_i = original_i + self.padding
        new_j = original_j + self.padding

        img_patch_t1 = self.img_t1_padding[new_i - self.padding:new_i + self.padding + 1,
                       new_j - self.padding:new_j + self.padding + 1, :].transpose(2, 0, 1)
        img_patch_t2 = self.img_t2_padding[new_i - self.padding:new_i + self.padding + 1,
                       new_j - self.padding:new_j + self.padding + 1, :].transpose(2, 0, 1)
        img_patch_t1 = torch.from_numpy(img_patch_t1)
        img_patch_t2 = torch.from_numpy(img_patch_t2)

        if self.dataset_name == 'China' or self.dataset_name == 'River':
            label = self.img_gt[original_i, original_j]
        elif self.dataset_name == 'BayArea' or self.dataset_name == 'Barbara' or self.dataset_name == 'Liyukou' or self.dataset_name == 'Hermiston' :
            label = self.img_gt[original_i, original_j] - 1.
        label = torch.from_numpy(np.array(label, dtype=np.float32))
        return img_patch_t1, img_patch_t2, label, torch.tensor((original_i, original_j), dtype=torch.long)


if __name__ == '__main__':
    random.seed(123)
    data_list = [i for i in range(100)]
    sampled_data = random.sample(data_list, 49)
    print(sampled_data)
    train_data = HSICD_Dataset(mode='train')
    val_data = HSICD_Dataset(mode='val')
    img_t1, img_t2, label, pos = train_data.__getitem__(1)
    print(label.dtype)
