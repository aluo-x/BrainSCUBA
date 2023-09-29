# BrainSCUBA

Here is the core code we use for our method.

It can be broadly divided into a couple of sections:
* fMRI encoder definition and training, this trained a linear layer that maps between the CLIP ViT-B/32 embedding and the voxel-wise activations
* Computation of the ViT-B/32 embeddings of the unpaired natural image set. You can find this code in extract_image_CLIP_ViT_B_32_embeddings
* Projection code, this projects the linear weights onto the space of natural images
* Sentence generation, this is done via the caption_inference.py file, which calls CLIPCap


Also included is a PDF of the appendix for convenience. We hope you can find this code helpful.