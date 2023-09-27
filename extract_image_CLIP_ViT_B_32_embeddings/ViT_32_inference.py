import torch
import clip
from PIL import Image
from torchvision import transforms
import os
# from torchvision.transforms imporInterpolationMode
import h5py
import numpy  as np
from dset import CustomImageDataset

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
model.eval()

all_folders = list(range(898,1211))
all_folders = [str(x).zfill(5) for x in all_folders]
all_folders = [os.path.join("./LAION-A", x) for x in all_folders]
from time import time
for cur_folder in all_folders:
    print(cur_folder)
    if os.path.exists(cur_folder):
        dataset = CustomImageDataset(cur_folder)
        dloader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=False, num_workers=16, pin_memory=True)
        offset = 0
        with torch.no_grad():
            with torch.inference_mode():
                total_np = []
                total_idx = []
                for i in dloader:
                    old_time = time()
                    z =  i["img"].to("cuda")
                    total_idx.append(i["my_idx"].numpy())
                    output = model.encode_image(z).cpu().numpy()
                    new_time = time()
                    total_np.append(output)
                final_np = np.concatenate(total_np, axis=0)
                final_idx = np.concatenate(total_idx)
                np.save("./embeddings/"+os.path.basename(cur_folder)+".npy", final_np)
                np.save("./embeddings/"+os.path.basename(cur_folder)+"_idx.npy", final_idx)
                del final_np
                del final_idx
                del dloader
                del dataset
