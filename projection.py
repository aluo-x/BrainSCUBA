import numpy as np
import matplotlib.pyplot as plt
import scipy
import torch
import os
import pickle

open_image_embeddings = np.load("./Open_Images_V7/embeddings/images.npy")
word_image_embeddings = np.concatenate([np.load("./Open_Images_V7/embeddings/train.npy"),np.load("./Open_Images_V7/embeddings/test.npy"),np.load("./Open_Images_V7/embeddings/validation.npy")])

total = word_image_embeddings.shape[0] + open_image_embeddings.shape[0]
open_sample = int(float(open_image_embeddings.shape[0])/2.0)
word_sample = int(float(word_image_embeddings.shape[0])/2.0)
other_sample = open_sample+word_sample

def get_files(raw_path):
    _files = os.listdir(raw_path)
    return [os.path.join(raw_path, _) for _ in _files]

possible_embeddings = get_files("./LAION-A/embeddings")
possible_embeddings = [_ for _ in possible_embeddings if (not ("idx" in _))]
possible_embeddings = sorted(possible_embeddings)

def retrive_embeddings(input_list, index):
    start = 0
    results = []
    for npname in input_list:
        # print(npname)
        value = np.load(npname)
        num_embeddings = value.shape[0]
        valid_index = index[np.logical_and(index>=start, index<num_embeddings)]
        if len(valid_index)>0:
            results.append(value[valid_index]+0.0)
        del value
        index = index-num_embeddings
    return np.concatenate(results, axis=0)


def softmax_convex(input_tensor, target_vector_directions, target_vector_norms, temp=100.0, use_torch = True, batch_size=10):
    # input_tensor = [nxdim], unit norm vectors
    # target_vector_directions = [kxdim], unit norm vecotrs
    # target_vector_norms = [kx1], scalars
    output_values = []
    if use_torch:
        # assume that all inputs are torch tensors
        with torch.no_grad():
            with torch.inference_mode():
                for k in range(0, len(input_tensor), batch_size):
                    normed_cur_batch = input_tensor[k:k+batch_size]
                    sims = normed_cur_batch@target_vector_directions.T*temp
                    scores = torch.nn.functional.softmax(sims,dim=-1)
                    soft_norms = scores@target_vector_norms
                    soft_projected = scores@target_vector_directions
                    soft_projected = soft_projected/torch.linalg.norm(soft_projected, dim=-1, keepdim=True)*soft_norms
                    output_values.append(soft_projected+0.0)
        return_array = torch.concatenate(output_values, dim=0).cpu()
        return return_array
    else:
        # assume that all inputs are numpy arrays
        for k in range(0, len(input_tensor), batch_size):
            normed_cur_batch = input_tensor[k:k+batch_size]
            sims = normed_cur_batch@target_vector_directions.T*temp
            scores = scipy.special.softmax(sims,axis=-1)
            
            soft_norms = scores@target_vector_norms
            soft_projected = scores@target_vector_directions
            soft_projected = soft_projected/np.linalg.norm(soft_projected, axis=-1, keepdims=True)*soft_norms
            output_values.append(soft_projected+0.0)
        return np.concatenate(output_values, axis=0)
            

np.random.seed(42)
rand_indices = np.random.choice(10469981, size=10000000-5*other_sample, replace=False)
rand_indices = np.split(rand_indices, 5)
container = {}
try:
    del inputs
except:
    pass
for subj in [1,2,5,7,3,4,6,8]:
    print("SUBJ", subj)
    inputs = torch.load("./S{}/00100.chkpt".format(subj), map_location="cpu")["network"]["linear.weight"].numpy()
    inputs = inputs/np.linalg.norm(inputs, axis=1, keepdims=True)
    inputs = torch.from_numpy(inputs).float().cuda()
    print(inputs.shape)
    import gc
    for z in range(5):
        word_embed_sample = np.random.choice(word_image_embeddings.shape[0], size=word_sample, replace=False)
        open_embed_sample = np.random.choice(open_image_embeddings.shape[0], size=open_sample, replace=False)
        my_word_embed = word_image_embeddings[word_embed_sample].astype(np.single)+0.0
        my_open_embed = open_image_embeddings[open_embed_sample].astype(np.single)+0.0
        gc.collect()
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.empty_cache()
        cur_indices = rand_indices[z]
        print(z)
        my_embeddings = np.concatenate([retrive_embeddings(possible_embeddings, cur_indices).astype(np.single), my_open_embed, my_word_embed])
        print(my_embeddings.shape, my_open_embed.shape, my_word_embed.shape)
        my_embeddings_norm = torch.from_numpy(np.linalg.norm(my_embeddings, axis=1, keepdims=True)).cuda()
        my_embeddings_direction = torch.from_numpy(my_embeddings).cuda()/my_embeddings_norm
        results = softmax_convex(inputs, my_embeddings_direction, my_embeddings_norm,temp=150.0, batch_size=4)
        container[z] = results.numpy()+0.0
        del my_embeddings_direction
        del my_embeddings_norm
        del my_embeddings
        del results
    import pickle
    with open("projection_dict_S{}_new_150.pkl".format(subj), "wb") as f:
        pickle.dump(container, f)