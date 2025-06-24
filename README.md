# BrainSCUBA: Fine-Grained Natural Language Captions of Visual Cortex Selectivity (ICLR 2024)
### Andrew F. Luo, Margaret M. Henderson, Michael J. Tarr, Leila Wehbe
#### ICLR 2024
This is the official code release for ***BrainSCUBA: Fine-Grained Natural Language Captions of Visual Cortex Selectivity*** (ICLR 2024).

[Paper link](https://arxiv.org/abs/2310.04420)

***TLDR:*** The purpose of this framework is to generate natural language captions of the semantic selectivity of each voxel.

### Abstract
Understanding the functional organization of higher visual cortex is a central focus in neuroscience. Past studies have primarily mapped the visual and semantic selectivity of neural populations using hand-selected stimuli, which may potentially bias results towards pre-existing hypotheses of visual cortex functionality. Moving beyond conventional approaches, we introduce a data-driven method that generates natural language descriptions for images predicted to maximally activate individual voxels of interest. Our method -- Semantic Captioning Using Brain Alignments ("BrainSCUBA") -- builds upon the rich embedding space learned by a contrastive vision-language model and utilizes a pre-trained large language model to generate interpretable captions. We validate our method through fine-grained voxel-level captioning across higher-order visual regions. We further perform text-conditioned image synthesis with the captions, and show that our images are semantically coherent and yield high predicted activations. Finally, to demonstrate how our method enables scientific discovery, we perform exploratory investigations on the distribution of "person" representations in the brain, and discover fine-grained semantic selectivity in body-selective areas. Unlike earlier studies that decode text, our method derives voxel-wise captions of semantic selectivity. Our results show that BrainSCUBA is a promising means for understanding functional preferences in the brain, and provides motivation for further hypothesis-driven investigation of visual cortex.

Here is the core code we use for our method.

It can be broadly divided into a couple of sections:
* fMRI encoder definition and training, this trained a linear layer that maps between the CLIP ViT-B/32 embedding and the voxel-wise activations
* Computation of the ViT-B/32 embeddings of the unpaired natural image set. You can find this code in extract_image_CLIP_ViT_B_32_embeddings
* Projection code, this projects the linear weights onto the space of natural images
* Sentence generation, this is done via the caption_inference.py file, which calls CLIPCap
