from .SegNetModules import *

class SegNet(nn.Module):

    def __init__(self, in_channels=3, depth=5, blocks=3, out_channels=1, start_filters=32, 
                 residual=True, concat=True):
        super(SegNet, self).__init__()

        if type(blocks) is not tuple:
            blocks = (blocks, blocks)

        # coarse stage
        self.encoder = Encoder(in_channels=in_channels, depth=depth, blocks=blocks[0],
                                    start_filters=start_filters, residual=residual, norm='bn',act=F.relu)
        
        self.decoder = Decoder(in_channels=start_filters * 2 ** (depth - 1),
                                out_channels=out_channels, depth=depth - 1,
                                blocks=blocks[1], residual=residual, 
                                concat=concat, norm='bn', use_att=True)

    def forward(self, x):
        image_code, before_pool = self.encoder(x)

        mask = self.decoder(image_code, before_pool) # list of 7 elements
        
        return mask
    
    def mask_prediction(self, x):
        image_code, before_pool = self.encoder(x)

        mask = self.decoder(image_code, before_pool)[0]
        mask = (mask > 0.5).float()

        return mask


class Encoder(nn.Module):
    def __init__(self, in_channels=3, depth=5, blocks=3, start_filters=32, residual=True, norm=nn.BatchNorm2d, act=F.relu):
        super(Encoder, self).__init__()
        self.down_convs = []
        outs = None
        for i in range(depth):
            ins = in_channels if i == 0 else outs
            outs = start_filters*(2**i)
            if i < depth - 1:
                pooling = True
            else: 
                pooling =  False
            down_conv = DownConv(ins, outs, blocks, pooling=pooling, residual=residual, norm=norm, act=act)
            self.down_convs.append(down_conv)
        self.down_convs = nn.ModuleList(self.down_convs)
        reset_params(self)

    def forward(self, x):
        encoder_outs = []
        for d_conv in self.down_convs:
            x, before_pool = d_conv(x)
            encoder_outs.append(before_pool)
        return x, encoder_outs
    

class Decoder(nn.Module):
    def __init__(self, in_channels=512, out_channels=1, norm='bn',act=F.relu, depth=5, blocks=3, residual=True,
                 concat=True, use_att=False):
        super(Decoder, self).__init__()
        self.up_convs_mask = []

        self.atts_mask = []
        self.use_att = use_att
        outs = in_channels
        for i in range(depth): 
            ins = outs
            outs = ins // 2
            
            up_conv = SMRBlock(ins, outs, blocks=blocks, residual=residual, concat=concat, norm=norm, act=act)
            self.up_convs_mask.append(up_conv)
            if self.use_att:
                self.atts_mask.append(ECABlock(outs))

        self.up_convs_mask = nn.ModuleList(self.up_convs_mask)
        self.atts_mask = nn.ModuleList(self.atts_mask)
        
        reset_params(self)

    def forward(self, mask_x, encoder_outs):
        mask_outs = []
        for i, up_mask in enumerate(self.up_convs_mask):
            before_pool = None
            if encoder_outs is not None:
                before_pool = encoder_outs[-(i+2)]
            if self.use_att:
                mask_before_pool = self.atts_mask[i](before_pool)
            smr_outs = up_mask(mask_x, mask_before_pool)
            mask_x= smr_outs['feats'][0]
            primary_map, self_calibrated_map = smr_outs['attn_maps']
            if i > 0:
                mask_outs.append(primary_map)
                mask_outs.append(self_calibrated_map)
            if i == len(self.up_convs_mask) - 1:
                mask_x = mask_outs[-1]

        return [mask_x] + mask_outs

    
