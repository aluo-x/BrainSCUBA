import argparse
import torch
import os
import random
import numpy as np

def bool_flag(s):
    if s == '1':
        return True
    elif s == '0':
        return False
    msg = 'Invalid value "%s" for bool flag (should be 0 or 1)'
    raise ValueError(msg % s)

def list_float_flag(s):
    return [float(_) for _ in list(s)]

class Options():

    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False
    # self.subject_id = arg_stuff.subject_id
    #         self.neural_activity_path = arg_stuff.neural_activity_path
    #         self.image_path = arg_stuff.image_path
    #         self.double_mask_path = ""
    def initialize(self):
        parser = self.parser
        parser.add_argument('--exp_name', default="subject_{}_ICLR")



        parser.add_argument('--save_loc', default="./results", type=str)
        parser.add_argument('--subject_id', default=["1"], nargs='+')
        parser.add_argument('--gpus', default=1, type=int) # Number of GPUs to use
        parser.add_argument('--neural_activity_path', default="/ROOT/data/cortex_subj_{}.npy")
        parser.add_argument('--image_path', default="/ROOT/data/image_data.h5py")
        parser.add_argument('--double_mask_path', default="/ROOT/double_mask_HCP.pkl")
        parser.add_argument('--volume_functional_path', default="/ROOT/volume_to_functional.pkl")
        parser.add_argument('--early_visual_path', default="/ROOT/rois/subj0{}/prf-visualrois.nii.gz")

        parser.add_argument('--epochs', default=100, type=int) # Total epochs to train for
        parser.add_argument('--resume', default=0, type=bool_flag) # Load weights or not from latest checkpoint
        parser.add_argument('--batch_size', default=32, type=int)
        parser.add_argument('--lr_init', default=3e-4, type=float)  # Starting learning rate for adam/adamw
        # parser.add_argument('--lr_init', default=5e-4, type=float)  # Starting learning rate for madgrad

        parser.add_argument('--lr_decay', default=5e-1, type=float)  # Learning rate decay rate


    def parse(self):
        # initialize parser
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        torch.manual_seed(0)
        # random.seed(0)
        np.random.seed(0)
        args = vars(self.opt)
        print('| options')
        for k, v in args.items():
            print('%s: %s' % (str(k), str(v)))
        print()
        return self.opt