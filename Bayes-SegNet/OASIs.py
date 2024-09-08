import os
from PIL import Image
from collections import OrderedDict
from torchvision.datasets import vision, utils
from typing import Any, Callable, List, Optional, Tuple
import json
import numpy as np

class OASIs(vision.VisionDataset):
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
        image_set: str = "type1",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)
        valid_image_sets = ["type1", "type2", "type3"]
        self.image_set = utils.verify_str_arg(image_set, "image_set", valid_image_sets)
        
        image_dir = os.path.join(root, f"{image_set}")
        target_dir = os.path.join(root, f"{image_set}_mask")
            
        self.images = [os.path.join(image_dir, x) for x in sorted(os.listdir(image_dir))]
        self.targets = [os.path.join(target_dir, x) for x in sorted(os.listdir(target_dir))]
        
        assert len(self.images) == len(self.targets)
    
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
