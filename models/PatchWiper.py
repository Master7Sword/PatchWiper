import torch
import torch.nn as nn
from .RestoreNetModules import RepresentationGenerationNet, PatchQueryNet
from .SegNet import SegNet

class PatchWiper(nn.Module):
    def __init__(self):
        super(PatchWiper, self).__init__()

        self.SegNet = SegNet()
        self.PQN = PatchQueryNet(num_outputs=3, width=64, depth=5)
        self.RGN = RepresentationGenerationNet(self.PQN.num_params, n_downsampling=2, dim=64)

    def forward(self, x):
        mask = self.SegNet.mask_prediction(x)
        input = torch.cat((x, mask), dim=1)
        
        params_4x = self.RGN(input)
        output = self.PQN(x, mask, params_4x)

        output = output + x 
        output = output * mask + x * (1 - mask)
        
        return output, mask

    def mask_prediction(self, x):
        mask = self.SegNet.mask_prediction(x)

        params, lr_mask = self.RGN.mask_prediction(x, mask)
        output = self.PQN.mask_prediction(x, mask, params, lr_mask)

        output = output * mask + x * (1 - mask)

        return output, mask