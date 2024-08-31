import torch.utils.data as data
from torchvision import transforms
import PIL.Image as Image
import os
import glob
import numpy as np
# from sklearn.model_selection import train_test_split
# from skimage.io import imread

    
class MaSTr1325Dataset(data.Dataset):
    def __init__(self, state, transform=None, target_transform=None):
        self.state = state
        self.root = "../Dataset/MaSTr1325"
        self.train_root = f"{self.root}/train"
        self.test_root  = f"{self.root}/test"
        self.val_root   = f"{self.root}/val"
        self.train_mask_root = f"{self.root}/train_mask"
        self.test_mask_root  = f"{self.root}/test_mask"
        self.val_mask_root   = f"{self.root}/val_mask"
        self.pics,self.masks = self.getDataPath()
        self.transform = transform
        self.target_transform = target_transform

    def getDataPath(self):
        assert self.state =='train' or self.state == 'val' or self.state =='test'
        # print(self.state)
        if self.state == 'train':
            img_root = self.train_root
            mask_root = self.train_mask_root
        if self.state == 'val':
            img_root = self.val_root
            mask_root = self.val_mask_root
        if self.state == 'test':
            img_root = self.test_root
            mask_root = self.test_mask_root

        # img_files = sorted(glob.glob(img_root + '/*.jpg'))
        # mask_files = sorted(glob.glob(mask_root + '/*.png')) 

        pics = []
        masks = []
    
        n = len(os.listdir(img_root))
        # print(n)
        for i in range(n):
            img = os.path.join(img_root, f"{i+1:03d}.jpg")
            mask = os.path.join(mask_root, f"{i+1:03d}m.png")
            pics.append(img)
            masks.append(mask)
            #imgs.append((img, mask))
        return pics,masks
    

    def __getitem__(self, index):
        #x_path, y_path = self.imgs[index]
        x_path = self.pics[index]
        y_path = self.masks[index]
        origin_x = Image.open(x_path).convert('RGB')
        origin_y = np.array(Image.open(y_path))
        origin_y = np.stack([origin_y==0, origin_y==1, origin_y==2], axis=-1).astype(np.float32)
        # print(f"{self.state} Dimension: {origin_y.ndim}")
        
        if self.transform is not None:
            img_x = self.transform(origin_x)
        if self.target_transform is not None:
            img_y = self.target_transform(origin_y)
        return img_x, img_y,x_path,y_path

    def __len__(self):
        return len(self.pics)
    
class OASIsDataset(data.Dataset):
    def __init__(self, state, transform=None, target_transform=None):
        self.state = state
        self.root = "../Dataset/OASIs"
        self.t1_root = f"{self.root}/type1"
        self.t2_root  = f"{self.root}/type2"
        self.t3_root   = f"{self.root}/type3"
        self.t1_mask_root = f"{self.root}/type1_mask"
        self.t2_mask_root  = f"{self.root}/type2_mask"
        self.t3_mask_root   = f"{self.root}/type3_mask"
        self.pics,self.masks = self.getDataPath()
        self.transform = transform
        self.target_transform = target_transform

    def getDataPath(self):
        assert self.state =='type1' or self.state == 'type2' or self.state =='type3'
        # print(self.state)
        if self.state == 'type1':
            img_root = self.t1_root
            mask_root = self.t1_mask_root
        if self.state == 'type2':
            img_root = self.t2_root
            mask_root = self.t2_mask_root
        if self.state == 'type3':
            img_root = self.t3_root
            mask_root = self.t3_mask_root

        # img_files = sorted(glob.glob(img_root + '/*.jpg'))
        # mask_files = sorted(glob.glob(mask_root + '/*.png')) 

        pics = []
        masks = []
    
        n = len(os.listdir(img_root))
        # print(n)
        for i in range(n):
            img = os.path.join(img_root, f"{i+1:03d}.jpg")
            mask = os.path.join(mask_root, f"{i+1:03d}m.png")
            pics.append(img)
            masks.append(mask)
            #imgs.append((img, mask))
        return pics,masks
    

    def __getitem__(self, index):
        #x_path, y_path = self.imgs[index]
        x_path = self.pics[index]
        y_path = self.masks[index]
        origin_x = Image.open(x_path).convert('RGB')
        origin_y = np.array(Image.open(y_path))
        origin_y = np.stack([origin_y==0, origin_y==1, origin_y==2], axis=-1).astype(np.float32)
        # print(f"{self.state} Dimension: {origin_y.ndim}")
        
        if self.transform is not None:
            img_x = self.transform(origin_x)
        if self.target_transform is not None:
            img_y = self.target_transform(origin_y)
        return img_x, img_y,x_path,y_path

    def __len__(self):
        return len(self.pics)

# # Example of how to use the DataLoader with this dataset
# if __name__ == "__main__":
#     x_transforms = transforms.Compose([
#         transforms.ToTensor(),
#     ])
#     y_transforms = transforms.Compose([
#         transforms.ToTensor(),
#     ])
    
#     train_dataset = MaSTr1325Dataset(state='train', transform=x_transforms, target_transform=y_transforms)
#     train_dataloader = data.DataLoader(train_dataset, batch_size=1, shuffle=True)

#     val_dataset = MaSTr1325Dataset(state='val', transform=x_transforms, target_transform=y_transforms)
#     val_dataloader = data.DataLoader(val_dataset, batch_size=1, shuffle=False)

#     test_dataset = MaSTr1325Dataset(state='test', transform=x_transforms, target_transform=y_transforms)
#     test_dataloader = data.DataLoader(test_dataset, batch_size=1, shuffle=False)

#     for batch_idx, (img_x, img_y, x_path, y_path) in enumerate(train_dataloader):
#         print(f"Batch {batch_idx + 1}")
#         print(f"Images: {x_path}")
#         print(f"Masks: {y_path}")
#         if batch_idx == 1:  # Print only first two batches for demonstration
#             break
