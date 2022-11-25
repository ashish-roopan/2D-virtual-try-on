import json
import cv2
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2


class cloth_dataset(Dataset):
    """cloth dataset.
    X : face_image + garment_image
    Y : bbox [1,4]"""

    def __init__(self, root_dir, transform=None, index=None):
        """
        Args:
            root_dir(straing): Directory with image and exp_parameter folders.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.og_height = 1101
        self.og_width = 750
        
        dirs = os.listdir(os.path.join(root_dir))
        self.face_images = [ os.path.join(dir, 'facehair.png') for dir in dirs]
        self.garment_images = [ os.path.join(dir, 'garment_top.png') for dir in dirs]
        self.labels = [ os.path.join(dir, 'positions.json') for dir in dirs]

        if index is not None:
            self.face_images = self.face_images[index[0]:index[1]]
            self.garment_images = self.garment_images[index[0]:index[1]]
            self.labels = self.labels[index[0]:index[1]]



    def __len__(self):
        return len(self.face_images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        #get face image
        face_img_name = os.path.join(self.root_dir, self.face_images[idx])
        face_image = cv2.imread(face_img_name, cv2.IMREAD_UNCHANGED)

        #get garment image
        garment_img_name = os.path.join(self.root_dir, self.garment_images[idx])
        garment_image = cv2.imread(garment_img_name, cv2.IMREAD_UNCHANGED)

        #hstack face and garment image
        face_image_bg = np.zeros_like(garment_image) 
        face_image_bg[0:face_image.shape[0], 0:face_image.shape[1]] = face_image
        full_image = np.hstack((face_image_bg, garment_image))

        
        # transform
        image = cv2.cvtColor(full_image, cv2.COLOR_BGR2RGB)
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']

        #get label (bbox)
        label_name = os.path.join(self.root_dir, self.labels[idx])
        position_data = json.load(open(label_name))
        x = position_data['x']
        y = position_data['y']
        w = position_data['w']
        h = position_data['h']
        position = torch.tensor([x,y,w,h])
        position = position/ torch.tensor([self.og_width, self.og_height, self.og_width, self.og_height])

        return full_image, image, position

  

    
def get_transforms():
    image_transforms = { 
        'train': A.Compose([
            A.Resize(256, 256),
            A.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225],),
            ToTensorV2()
        ]),

        'valid': A.Compose([
            A.Resize(256, 256),
            A.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225],),
            ToTensorV2()
            ]),


        'test': A.Compose([
            A.Resize(256, 256),
            A.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225],),
            ToTensorV2()
            ])
    }

    return image_transforms

def get_dataloader(data_dir, batch_size, split, num_images=None):
    # Load the Data
    image_transforms = get_transforms()

    #select how many images to train with. To overfit on 1 image, set num_images = 1
    if num_images is not None: 
        index = [0, int(num_images)]
    else:
        index = None

    dataset = cloth_dataset(root_dir=data_dir, transform=image_transforms[split], index=index)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=10)

    return dataloader
