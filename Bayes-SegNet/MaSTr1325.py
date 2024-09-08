import os
from PIL import Image
from collections import OrderedDict
from torchvision.datasets import vision, utils
from typing import Any, Callable, List, Optional, Tuple
import json
import numpy as np

class MaSTr1325(vision.VisionDataset):
    color_encoding = OrderedDict([
        ('obstacles_and_environment', (255, 255, 0)),
        ('water', (0, 133, 152)),
        ('sky', (129, 0, 255)),
        ('unknown', (255, 0, 0)),
    ])
    
    # mapping original labels to continuous labels
    label_mapping = {
        0: 0,  # obstacles_and_environment
        1: 1,  # water
        2: 2,  # sky
        4: 3,  # unknown
    }
    
    def __init__(
        self,
        root: str,
        image_set: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)
        valid_image_sets = ["train", "val", "test"]
        self.image_set = utils.verify_str_arg(image_set, "image_set", valid_image_sets)
        
        image_dir = os.path.join(root, f"{image_set}")
        target_dir = os.path.join(root, f"{image_set}_mask")
            
        self.images = [os.path.join(image_dir, x) for x in sorted(os.listdir(image_dir))]
        self.targets = [os.path.join(target_dir, x) for x in sorted(os.listdir(target_dir))]
        
        assert len(self.images) == len(self.targets)

        # # Save in json files to check mapping
        # with open('images.json', 'w') as f:
        #     json.dump(self.images, f, indent=4)
        # with open('targets.json', 'w') as f:
        #     json.dump(self.targets, f, indent=4)
    
    def __len__(self) -> int:
        return len(self.images)
    
    @property
    def masks(self) -> List[str]:
        return self.targets
    
    def map_labels(self, target: Image.Image) -> Image.Image:
        target = np.array(target) # convert into numpy format
        mapped_target = np.where(target == 4, 3, target) # remapping
        return Image.fromarray(mapped_target.astype(np.uint8))
        
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        img = Image.open(self.images[index]).convert("RGB")
        target = Image.open(self.masks[index])     
        
        # mapping the original labels to continuous labels
        target = self.map_labels(target)

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

        
if __name__ == "__main__":
    #%%
    import torch
    from utils import LocalContrastNormalisation, LongTensorToRGBPIL, PILToLongTensor, batch_transform, imshow_batch
    from torch.utils.data import DataLoader
    from torchvision import transforms
    from PIL import Image

    transform = transforms.Compose([
        transforms.Resize((384, 512)),
        transforms.ToTensor(),
        LocalContrastNormalisation(3, 0, 1)
    ])

    target_transform = transforms.Compose([
        transforms.Resize((384, 512)),
        PILToLongTensor()
    ])

    root = os.path.expanduser("~/BDL/BDL-based-semantic-segmentation/Dataset/MaSTr1325")
    print(f"root path: {root}")
    train_data = MaSTr1325(root, "train", 
                        transform=transform, target_transform=target_transform)
    train_loader = DataLoader(train_data, 5)

    # Get encoding between pixel valus in label images and RGB colors
    class_encoding = train_data.color_encoding
    # print(class_encoding)
    # OrderedDict([
    # ('obstacles_and_environment', (255, 255, 0)), 
    # ('water', (0, 133, 152)), 
    # ('sky', (129, 0, 255)), 
    # ('unknown', (255, 0, 0))
    # ])

    # Get number of classes to predict
    num_classes = len(class_encoding)
    print(f"The number of classes: {num_classes}")

    # Get a batch of samples to display
    images, labels = next(iter(train_loader))
    print(images.shape, labels.shape)
    print(images.min(), images.max())
    # assert False

    # Show a batch of samples and labels
    print("Close the figure window to continue...")
    label_to_rgb = transforms.Compose([
        # transforms.ToPILImage(),
        LongTensorToRGBPIL(class_encoding),
        transforms.ToTensor()
    ])
    color_labels = batch_transform(labels, label_to_rgb)
    imshow_batch(images, color_labels)


    # print(train_data[0][0].shape)
    # imgs = torch.stack([img_t for img_t, _ in train_data], dim=3)
    # print(torch.mean(imgs.reshape(3, -1), dim=-1), torch.std(imgs.reshape(3, -1), dim=-1))