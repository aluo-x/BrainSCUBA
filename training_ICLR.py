import torch

torch.backends.cudnn.benchmark = True
import sys
import os

sys.path.append("./EVA/EVA-CLIP/rei")
os.environ["HF_HOME"] = "./cache"

from data_loader_224 import neural_loader
import torch.multiprocessing as mp
import os
import socket
from torch.cuda.amp import GradScaler
from contextlib import closing
import torch.distributed as dist
import model
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
import math
from time import time
from options import Options
import functools
import random
from torch import autocast
# from eva_clip import create_model_and_transforms, get_tokenizer
import clip
import timm
torch.backends.cudnn.benchmark=True

def find_free_port():
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('localhost', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]

def worker_init_fn(worker_id, myrank_info):
    # print(worker_id + myrank_info*100, "SEED")
    np.random.seed(worker_id + myrank_info*100)

def shuffle_shift(input_image, extent=4):
    offset_x = random.randint(-extent, extent)
    offset_y = random.randint(-extent, extent)
    orig_shape = input_image.shape
    temp = input_image[:,:, max(0,offset_x):min(orig_shape[2], orig_shape[2]+offset_x), max(0,offset_y):min(orig_shape[3], orig_shape[3]+offset_y)]
#     temp = torch.nn.functional.pad(temp, (max(0,offset_y), max(0, -offset_y), max(0,offset_x), max(0, -offset_x)), mode='replicate')
    temp = torch.nn.functional.pad(temp, (max(0, -offset_y),max(0,offset_y), max(0, -offset_x), max(0,offset_x)), mode='replicate')
    return temp

def train_net(rank, world_size, freeport, other_args):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = freeport
    output_device = rank
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    torch.backends.cudnn.benchmark = True
    # other_args.subject_id = [1,2,3,4,5,6,7,8]
    dataset = neural_loader(other_args)
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    ranked_worker_init = functools.partial(worker_init_fn, myrank_info=rank)
    neural_dataloader = torch.utils.data.DataLoader(dataset, batch_size=other_args.batch_size//world_size, shuffle=False, num_workers=4, worker_init_fn=ranked_worker_init, persistent_workers=True, sampler=train_sampler,drop_last=False)
    print(dataset.early_sizes, dataset.higher_sizes, "SIZES")
    # dist.barrier()
    # dist.destroy_process_group()
    # exit()

    model_name = "EVA02-CLIP-B-16"
    pretrained = "eva_clip"  # or "/path/to/EVA02_CLIP_B_psz16_s8B.pt"
    # feature_extractor, _, __ = create_model_and_transforms(model_name, pretrained, force_custom_clip=True)
    # feature_extractor = timm.create_model('vit_base_patch16_clip_224.laion2b', pretrained=True)
    feature_extractor, _ = clip.load("ViT-B/32", device="cuda")

    del feature_extractor.transformer
    torch.cuda.empty_cache()
    feature_extractor.eval()
    # projector = model_vit.downproject_split(num_early_output=dataset.early_sizes, num_higher_output=dataset.higher_sizes) # used intermediate + last CLIP layer
    # projector = model.soft_quantizer_v2(num_higher_output=dataset.higher_sizes[str(other_args.subject_id[0])])
    projector = model.soft_quantizer_v3(num_higher_output=dataset.higher_sizes[str(other_args.subject_id[0])])

    projector.train()
    print(projector.training, "TRAINING STATUS")
    if rank == 0:
        print("Dataloader requires {} batches".format(len(neural_dataloader)))

    start_epoch = 1
    load_opt = 0
    loaded_weights = False
    if other_args.resume:
        if not os.path.isdir(other_args.exp_dir):
            print("Missing save dir, exiting")
            dist.barrier()
            dist.destroy_process_group()
            return 1
        else:
            current_files = sorted(os.listdir(other_args.exp_dir))
            if len(current_files)>0:
                latest = current_files[-1]
                start_epoch = int(latest.split(".")[0]) + 1
                if rank == 0:
                    print("Identified checkpoint {} with new starting epoch {}".format(latest, start_epoch))
                if start_epoch >= (other_args.epochs+1):
                    dist.barrier()
                    dist.destroy_process_group()
                    return 1
                # map_location = 'cuda:%d' % rank
                map_location = 'cpu'
                weight_loc = os.path.join(other_args.exp_dir, latest)
                weights = torch.load(weight_loc, map_location=map_location)
                if rank == 0:
                    print("Checkpoint loaded {}".format(weight_loc))
                dist.barrier()
                projector.load_state_dict(weights["network"])
                loaded_weights = True
                if "opt" in weights:
                    load_opt = 1
                dist.barrier()
        if loaded_weights is False:
            print("Resume indicated, but no weights found!")
            dist.barrier()
            dist.destroy_process_group()
            exit()
    _ = projector.to(rank)
    # We have conditional forward, must set find_unused_parameters to true
    ddp_projector = DDP(projector, find_unused_parameters=False, device_ids=[rank], gradient_as_bucket_view=True)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(ddp_projector.parameters(), lr=other_args.lr_init, weight_decay=1.5e-2)
    print("USING AdamW with dropout double variant, DROP CONV, big decay")

    if load_opt:
        print("loading optimizer")
        optimizer.load_state_dict(weights["opt"])
        dist.barrier()

    if rank == 0:
        old_time = time()

    for epoch in range(start_epoch, other_args.epochs+1):
        decay_rate = other_args.lr_decay
        new_lrate = other_args.lr_init * (decay_rate ** (epoch / other_args.epochs))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate

        total_losses = 0
        cur_iter = 0

        train_sampler.set_epoch(epoch)
        for data_stuff in neural_dataloader:
            # with torch.no_grad():
            # start = torch.cuda.Event(enable_timing=True)
            # end = torch.cuda.Event(enable_timing=True)
            #
            # start.record()

            neural_data = data_stuff["neural_data"].to(output_device, non_blocking=True) # Flat tensor already
            image_data = data_stuff["image_data"][:,0].to(output_device, non_blocking=True) # collapse along batch
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                    features = feature_extractor.encode_image(shuffle_shift(image_data)+torch.randn_like(image_data)*0.05)
                    features = features/features.norm(dim=-1, keepdim=True)
            predicted = ddp_projector(features.float())
            # print(predicted.shape, neural_data.shape)
            loss = criterion(predicted, neural_data)
            if rank==0:
                total_losses += loss.detach()
                cur_iter += 1
                # if cur_iter % 50 == 0:
                #     print(loss.detach().item())
            loss.backward()
            optimizer.step()

        if rank == 0:
            avg_loss = total_losses.item() / cur_iter
            print("{}: Ending epoch {}, loss {}, time {}, lr {}".format(other_args.exp_name, epoch, avg_loss, time() - old_time, new_lrate))
            old_time = time()
        if rank == 0 and (epoch%20==0 or epoch==1 or epoch>(other_args.epochs-3)):
            save_name = str(epoch).zfill(5)+".chkpt"
            save_dict = {}
            save_dict["network"] = ddp_projector.module.state_dict()
            torch.save(save_dict, os.path.join(other_args.exp_dir, save_name))
        dist.barrier()
    print("Wrapping up training {}".format(other_args.exp_name))
    dist.barrier()
    dist.destroy_process_group()
    return 1


if __name__ == '__main__':
    cur_args = Options().parse()
    cur_args.exp_name = "subject_{}_linear"

    print("ICLR CLIP")

    exp_name = cur_args.exp_name
    if len(cur_args.subject_id[0])>1:
        cur_args.subject_id = sorted([str(int(sbjid)) for sbjid in cur_args.subject_id[0].split(",")])
    exp_name_filled = exp_name.format("-".join(cur_args.subject_id))
    cur_args.exp_name = exp_name_filled
    if not os.path.isdir(cur_args.save_loc):
        print("Save directory {} does not exist, creating...".format(cur_args.save_loc))
        os.mkdir(cur_args.save_loc)
    exp_dir = os.path.join(cur_args.save_loc, exp_name_filled)
    cur_args.exp_dir = exp_dir
    print("Experiment directory is {}".format(exp_dir))
    if not os.path.isdir(exp_dir):
        os.mkdir(exp_dir)
    world_size = cur_args.gpus
    myport = str(find_free_port())
    mp.spawn(train_net, args=(world_size, myport, cur_args), nprocs=world_size, join=True)