
import torch
import torch.nn as nn

from csi_reference.CRNet.utils import logger
from csi_reference.CRNet.my_net import *

__all__ = ["mynet"]

class MyNet(nn.Module):
    def __init__(self, args):
        super(MyNet, self).__init__()
        total_size = args.last_channels * 1024
        self.last_channels = args.last_channels
        
        self.encode_model = Balle2Encoder(args.channels, args.last_channels, args.norm)
        self.quant_noise = NoiseQuant(table_range=args.table_range)
        self.quant_ste = SteQuant(table_range=args.table_range)
        self.cit_he = TransHyperScale(cin=args.last_channels, cout=args.hyper_channels, scale=args.scale, down=True, opt=args)
        self.cit_hd = TransHyperScale(cin=args.hyper_channels, scale=args.scale, down=False, opt=args)
        self.cit_ar = TransDecoder(cin=args.last_channels, opt=args)
        
        self.cit_pn = torch.nn.Sequential(
            nn.Conv2d(args.dim_embed*2, args.dim_embed*args.mlp_ratio, 1, 1, 0),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(args.dim_embed*args.mlp_ratio, args.last_channels*args.K*args.num_parameter, 1, 1, 0),
        )
        
        self.encoder_fc = nn.Linear(total_size//4, total_size // args.cr)
        self.decoder_fc = nn.Linear(total_size // args.cr, total_size//4)
        
        self.decode_model = Balle2Decoder(args.channels, args.last_channels, args.norm)
        self.prob_model = Entropy(args.hyper_channels)
        
    def forward(self, x):
        n, c, h, w = x.detach().size()

        y = self.encode_model(x*2 - 1)
        # y = self.encoder_fc(y.view(n, -1)).view(n, self.last_channels, 8, -1)
        y_tilde = self.quant_noise(y)
        y_tilde2 = self.quant_ste(y)   # quant_ste(y), y_tilde
        # Hyperprio Transformer Entropy Model
        z = self.cit_he(y)
        z_tilde = self.quant_noise(z)
        feat_hyper = self.cit_hd(z_tilde)
        # Auto-regressive Transformer Entropy Model
        feat_ar = self.cit_ar(y_tilde)
        # Merge 2 features and Parameter Network
        feat_merge = torch.cat([feat_hyper,feat_ar], 1)
        predicted_param = self.cit_pn(feat_merge)      
        # Decoder
        # y_tilde2 = self.decoder_fc(y_tilde2.view(n, -1)).view(n, self.last_channels, h, w)
        
        y_tilde2 = self.encoder_fc(y_tilde2.view(n, -1))
        y_tilde2 = self.decoder_fc(y_tilde2).view(n, c, h//2, w//2)
        
        x_tilde = self.decode_model(y_tilde2)
        x_tilde = x_tilde / 2 + 0.5
        x_tilde = torch.clamp(x_tilde, 0., 1.)
        # Probability model of hyperprior information
        z_prob = self.prob_model(z_tilde)
        
        return x_tilde, y_tilde, z_prob, predicted_param
        
        
def mynet(args):
    model = MyNet(args)
    return model