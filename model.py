import torch

class soft_quantizer_v3(torch.nn.Module):
    def __init__(self, num_higher_output=1000, num_prototypes=4, embed_dim=512, clip_offset=5.0, max_norm=10.0):
        super().__init__()
        self.linear = torch.nn.Linear(embed_dim, num_higher_output)
    def forward(self, image_vectors):
        # image_vectors should be pre_normalized
        return self.linear(image_vectors)
