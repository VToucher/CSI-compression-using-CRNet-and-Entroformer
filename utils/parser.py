import argparse

parser = argparse.ArgumentParser(description='CRNet PyTorch Training')


# ========================== Indispensable arguments ==========================

parser.add_argument('--data_dir', type=str, required=False, default=None,
                    help='the path of dataset.')
parser.add_argument('--scenario', type=str, required=False, choices=["in", "out"], default='in',
                    help="the channel scenario")
parser.add_argument('-b', '--batch_size', type=int, required=False, metavar='N', default=200,
                    help='mini-batch size')
parser.add_argument('-j', '--workers', type=int, metavar='N', required=False, default=0,
                    help='number of data loading workers')


# ============================= Optical arguments =============================

# Working mode arguments
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', type=str, default=None,
                    help='using locally pre-trained model. The path of pre-trained model should be given')
parser.add_argument('--resume', type=str, metavar='PATH', default=None,
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--cpu', action='store_true',
                    help='disable GPU training (default: False)')
parser.add_argument('--cpu-affinity', default=None, type=str,
                    help='CPU affinity, like "0xffff"')

# Other arguments
parser.add_argument('--epochs', type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--cr', metavar='N', type=int, default=4,
                    help='compression ratio')
parser.add_argument('--scheduler', type=str, default='const', choices=['const', 'cosine'],
                    help='learning rate scheduler')

# ============================= Entrotransformer arguments =============================
# Base configure
parser.add_argument("--channels", type=int, default=16, 
                    help="Channels in Main Auto-encoder.")
parser.add_argument("--last_channels", type=int, default=32, 
                    help="Channels of compression feature.")
parser.add_argument("--hyper_channels", type=int, default=16, 
                    help="Channels of hyperprior feature.")
parser.add_argument("--norm", type=str, default="GDN", 
                    help="Normalization Type: GDN, GSDN")
parser.add_argument("--table_range", type=int, default=4, 
                    help="range of feature")
parser.add_argument("--K", type=int, default=1, 
                    help="the number of Mix Hyperprior.")
parser.add_argument("--num_parameter", type=int, default=2,
                    help="distribution parameter num: 1 for sigma, 2 for mean&sigma, 3 for mean&sigma&pi")
parser.add_argument("--alpha", type=float, default=0.01, 
                    help="weight for reconstruction loss")

# Configure for Transfomer Entropy Model
parser.add_argument("--dim_embed", type=int, default=16, 
                    help="Dimension of transformer embedding.")
parser.add_argument("--depth", type=int, default=6, 
                    help="Depth of CiT.")
parser.add_argument("--heads", type=int, default=6, 
                    help="Number of transformer head.")
parser.add_argument("--mlp_ratio", type=int, default=4, 
                    help="Ratio of transformer MLP.")
parser.add_argument("--dim_head", type=int, default=6, 
                    help="Dimension of transformer head.")
parser.add_argument("--trans_no_norm", dest="trans_norm", action="store_false", default=True, 
                    help="Use LN in transformer.")
parser.add_argument("--dropout", type=float, default=0., 
                    help="Dropout ratio.")
parser.add_argument("--position_num", type=int, default=7, 
                    help="Position information num.")
parser.add_argument("--att_noscale", dest="att_scale", action="store_false", default=True, 
                    help="Use Scale in Attention.")
parser.add_argument("--no_rpe_shared", dest="rpe_shared", action="store_false", default=True, 
                    help="Position Shared in layers.")
parser.add_argument("--scale", type=int, default=1, 
                    help="Downscale of hyperprior of CiT.")
parser.add_argument("--mask_ratio", type=float, default=0., 
                    help="Pretrain model: mask ratio.")
parser.add_argument("--attn_topk", type=int, default=4, 
                    help="Top K filter for Self-attention.")    
parser.add_argument("--grad_norm_clip", type=float, default=1.0, 
                    help="grad_norm_clip.")
parser.add_argument("--warmup", type=float, default=0.05, 
                    help="Warm up.")
parser.add_argument("--segment", type=int, default=1, 
                    help="Segment for Large Patchsize.")    

# Training and testing configure
parser.add_argument("--patchSize", type=int, default=32, 
                    help="Training Image size.")


args = parser.parse_args()
