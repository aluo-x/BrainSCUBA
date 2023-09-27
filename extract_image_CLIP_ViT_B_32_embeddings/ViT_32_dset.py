import torch
import clip
from PIL import Image
from torchvision import transforms
import os
# from torchvision.transforms imporInterpolationMode
import h5py
import numpy  as np
from torch.utils.data import Dataset
def _convert_image_to_rgb(image):
    return image.convert("RGB")

class CustomImageDataset(Dataset):
    def __init__(self, folder):
        files_list = os.listdir(folder)
        files = [os.path.join(folder, x) for x in files_list]
        files = sorted(files)
        filtered_files = []
        for i in files:
            if (".jpg" in i) or (".png" in i) or (".jpeg" in i):
                filtered_files.append(i)
        self.files = filtered_files
        self.preprocess2 = transforms.Compose([transforms.Resize(size=224, interpolation=transforms.InterpolationMode.BICUBIC, max_size=None),
                                  transforms.CenterCrop(224),
                                 _convert_image_to_rgb,
                                 transforms.ToTensor(),
                                 transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))])
        

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        cur_file = self.files[idx]
        try:
            myimage = Image.open(cur_file, mode="r")
            img = self.preprocess2(myimage)
            myimage.close()
            my_idx = idx
        except Exception as e:
            print(e)
            img = torch.zeros(3,224,224)
            my_idx = -1
        return {"img":img, "my_idx":my_idx}
        
        
