from math import pi
from .transposed_attention import *


def _get_coords(bs, h, w, device, ds):
    """Creates the position encoding for the pixel-wise MLPs"""
    x = torch.arange(0, w).float()
    y = torch.arange(0, h).float()
    scale = 7 / 8

    x_cos = torch.remainder(x, ds).float() / ds
    x_sin = torch.remainder(x, ds).float() / ds
    y_cos = torch.remainder(y, ds).float() / ds
    y_sin = torch.remainder(y, ds).float() / ds

    x_cos = x_cos / (max(x_cos) / scale)
    x_sin = x_sin / (max(x_sin) / scale)
    y_cos = y_cos / (max(y_cos) / scale)
    y_sin = y_sin / (max(y_sin) / scale) 

    xcos = torch.cos((2 * pi * x_cos).float())
    xsin = torch.sin((2 * pi * x_sin).float())
    ycos = torch.cos((2 * pi * y_cos).float())
    ysin = torch.sin((2 * pi * y_sin).float())

    xcos = xcos.view(1, 1, 1, w).repeat(bs, 1, h, 1)
    xsin = xsin.view(1, 1, 1, w).repeat(bs, 1, h, 1)
    ycos = ycos.view(1, 1, h, 1).repeat(bs, 1, 1, w)
    ysin = ysin.view(1, 1, h, 1).repeat(bs, 1, 1, w)

    coords = torch.cat([xcos, xsin, ycos, ysin], 1).to(device)

    return coords.to(device)


class RepresentationGenerationNet(nn.Module):
    def __init__(self, 
                 num_out, 
                 dim=48,
                 n_downsampling=2, 
                 num_blocks = [4,6,6,8], 
                 heads = [1,2,4,8],
                 ffn_expansion_factor = 2.66,
                 bias = False,
                 LayerNorm_type = 'WithBias',
        ):

        super().__init__()
        fm_size = dim*(2**n_downsampling)

        self.encoder = OverlapPatchEmbed(in_c=4, embed_dim=dim, bias=True)

        downsample = []
        for i in range(n_downsampling):
            for _ in range(num_blocks[i]):
                downsample.append(TransformerBlock(dim=int(dim*2**i), num_heads=heads[i], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type))
            downsample.append(Downsample(dim*2**i))
        for _ in range(num_blocks[i+1]):
            downsample.append(TransformerBlock(dim=int(dim*2**(i+1)), num_heads=heads[i+1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type))
        if n_downsampling == 2:
            for _ in range(num_blocks[i+2]):
                downsample.append(TransformerBlock(dim=int(dim*2**(i+1)), num_heads=heads[i+1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type))
        self.downsample = nn.Sequential(*downsample)

        self.out_para = nn.Sequential(nn.Linear(fm_size, num_out))


    def mask_prediction(self, x, mask):
        input = torch.cat((x, mask), dim=1)
        features = self.encoder(input)

        features = self.downsample(features)
        features = features.permute(0, 2, 3, 1) # [bs, h/4, w/4, 256]

        bs, h, w, c = features.size()
        k = x.size(2) // h  # 4
        mask = mask.unfold(2, k, k).unfold(3, k, k) # [bs, 1, 256, 256] → [bs, 1, 64, 64, 4, 4]
        mask = mask.permute(0, 2, 3, 4, 5, 1).contiguous().view(bs, h, w, int(k*k)) # [bs, 64, 64, 16]
        lr_mask = torch.mean(mask, dim=-1).view(bs, h * w) # [bs, 4096]
        features = features.view(bs, h*w, c) # [bs, h*w/16, 256]

        index = torch.nonzero(lr_mask)[:, 1]  # 取所有行的第1列（列索引）
        
        features = features[:, index, :] # [bs, len(index), 256]
        output_params = self.out_para(features) # [bs, len(index), num_params]
        output_params = output_params.permute(0, 2, 1)

        return output_params, index
    
    def forward(self, x):
        features = self.encoder(x)

        features = self.downsample(features)

        features = features.permute(0, 2, 3, 1)

        output_params = self.out_para(features)
        output_params = output_params.permute(0, 3, 1, 2)

        return output_params
    

class PatchQueryNet(torch.nn.Module):
    def __init__(self, num_outputs=3, width=64, depth=5):
        super(PatchQueryNet, self).__init__()

        self.num_outputs = num_outputs
        self.width = width
        self.depth = depth
        self.xy_coords = None

        self.channels = []
        self._set_channels()

        self.num_params = 0
        self.splits = {}
        self._set_num_params()

    def _set_channels(self):
        in_ch = int(4) + int(3) + int(1) # 4: position, 3: RGB, 1: mask
        # in_ch = int(4)
        # in_ch = int(7)
        # in_ch = int(5)

        self.channels = [in_ch]
        for _ in range(self.depth - 1): 
            self.channels.append(self.width)

        self.channels.append(self.num_outputs)

    def _set_num_params(self):
        nparams = 0
        self.splits = {
            "biases": [],
            "weights": [],
        }
        idx = 0
        for layer, nci in enumerate(self.channels[:-1]):
            nco = self.channels[layer + 1]
            nparams = nparams + nco  
            self.splits["biases"].append((idx, idx + nco))
            idx = idx + nco
            nparams = nparams + nci * nco 
            self.splits["weights"].append((idx, idx + nco * nci))
            idx = idx + nco * nci

        self.num_params = nparams

    def _get_weight_indices(self, idx):
        return self.splits["weights"][idx]

    def _get_bias_indices(self, idx):
        return self.splits["biases"][idx]

    def forward(self, image_wm, mask, params): 
        assert params.shape[1] == self.num_params, "incorrect input params"

        bs, _, h, w = image_wm.shape  # h = w = 256 
        bs, _, h_lr, w_lr = params.shape  # h_lr = w_lr = 64
        k = h // h_lr  # k = 4

        self.xy_coords = _get_coords(1, h, w, image_wm.device, h // h_lr)

        input = torch.repeat_interleave(self.xy_coords, repeats=bs, dim=0) # (1, 4, h, w) 扩成 (batchsize, 4, h, w)
        input = torch.cat((input, image_wm, mask), dim=1) # (batchsize, 8, h, w)
        # input = torch.cat((input, image_wm), dim=1) # (batchsize, 7, h, w)
        # input = torch.cat((input, mask), dim=1) # (batchsize, 5, h, w)

        nci = input.shape[1] # 8

        tiles = input.unfold(2, k, k).unfold(3, k, k)
        tiles = tiles.permute(0, 2, 3, 4, 5, 1).contiguous().view(
            bs, h_lr, w_lr, int(k * k), nci)  # (bs, 64, 64, 16, 8)
        out = tiles
        num_layers = len(self.channels) - 1

        for idx, nci in enumerate(self.channels[:-1]):
            nco = self.channels[idx + 1]

            bstart, bstop = self._get_bias_indices(idx)
            wstart, wstop = self._get_weight_indices(idx)

            w_ = params[:, wstart:wstop]
            b_ = params[:, bstart:bstop]

            w_ = w_.permute(0, 2, 3, 1).view(bs, h_lr, w_lr, nci, nco)
            b_ = b_.permute(0, 2, 3, 1).view(bs, h_lr, w_lr, 1, nco)

            out = torch.matmul(out, w_) + b_

            if idx < num_layers - 1:
                out = torch.nn.functional.leaky_relu(out, 0.01)
            else:
                out = torch.tanh(out)
        
        out = out.view(bs, h_lr, w_lr, k, k, self.num_outputs).permute(
            0, 5, 1, 3, 2, 4)
        out = out.contiguous().view(bs, self.num_outputs, h, w)

        return out


    def mask_prediction(self, image_wm, mask, params, index): 
        assert params.shape[1] == self.num_params, "incorrect input params"

        # Fetch sizes
        bs, _, h, w = image_wm.shape  # h = w = 256 / 512 / 1024
        h_lr, w_lr = h//4, w//4
        k = 4

        self.xy_coords = _get_coords(1, h, w, image_wm.device, h // h_lr)

        input = torch.repeat_interleave(self.xy_coords, repeats=bs, dim=0) # (1, 4, h, w) 扩成 (batchsize, 4, h, w)
        input = torch.cat((input, image_wm, mask), dim=1) # (batchsize, 8, h, w)   
        # input = torch.cat((input, mask), dim=1) # (batchsize, 5, h, w)   

        nci = input.shape[1] # 8

        tiles = input.unfold(2, k, k).unfold(3, k, k)
        tiles = tiles.permute(0, 2, 3, 4, 5, 1).contiguous().view(
            bs, h_lr, w_lr, int(k * k), nci)  # (bs, 64, 64, 16, 8)

        out = tiles
        num_layers = len(self.channels) - 1

        out = out.view(bs, h_lr * w_lr, int(k * k), nci)[:, index, :, :]
        num = out.size(1)

        for idx, nci in enumerate(self.channels[:-1]):
            nco = self.channels[idx + 1]

            bstart, bstop = self._get_bias_indices(idx)
            wstart, wstop = self._get_weight_indices(idx)

            w_ = params[:, wstart:wstop]
            b_ = params[:, bstart:bstop]

            w_ = w_.permute(0, 2, 1).view(bs, num, nci, nco)
            b_ = b_.permute(0, 2, 1).view(bs, num, 1, nco)

            out = torch.matmul(out, w_) + b_

            if idx < num_layers - 1:
                out = torch.nn.functional.leaky_relu(out, 0.01)
            else:
                out = torch.tanh(out)
        
        image_wm = image_wm.unfold(2, k, k).unfold(3, k, k)
        image_wm = image_wm.permute(0, 2, 3, 4, 5, 1).contiguous().view(
                  bs, h_lr, w_lr, int(k * k), 3).view(bs, h_lr * w_lr, int(k * k), 3)
        
        image_wm[:, index, :, :] += out # 把输出paste到原图上
        out = image_wm.view(bs, h_lr, w_lr, k, k, self.num_outputs).permute(
            0, 5, 1, 3, 2, 4)
        out = out.contiguous().view(bs, self.num_outputs, h, w)

        return out
