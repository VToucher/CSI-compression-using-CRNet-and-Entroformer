import os
import random
import thop
import torch

from csi_reference.CRNet.models import crnet
from csi_reference.CRNet.models.mynet import mynet
from csi_reference.CRNet.utils import logger, line_seg

__all__ = ["init_device", "init_model"]


def init_device(seed=None, cpu=None, gpu=None, affinity=None):
    # set the CPU affinity
    if affinity is not None:
        os.system(f'taskset -p {affinity} {os.getpid()}')

    # Set the random seed
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

    # Set the GPU id you choose
    if gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)

    # Env setup
    if not cpu and torch.cuda.is_available():
        device = torch.device('cuda')
        torch.backends.cudnn.benchmark = True
        if seed is not None:
            torch.cuda.manual_seed(seed)
        pin_memory = True
        logger.info("Running on GPU%d" % (gpu if gpu else 0))
    else:
        pin_memory = False
        device = torch.device('cpu')
        logger.info("Running on CPU")

    return device, pin_memory


def init_model(args):
    # Model loading
    model = crnet(reduction=args.cr)
    
    if args.pretrained is not None:
        assert os.path.isfile(args.pretrained)
        state_dict = torch.load(args.pretrained,
                                map_location=torch.device('cpu'))['state_dict']
        # 验收添加
        # result_dict = {}
        # for key, weight in state_dict.items():
        #     result_key = key
        #     if key[:7] == "module.":
        #         result_key = key[7:]
        #     # if 'relative_position_index' not in key and 'relative_position_bias_table' not in key:
        #     if "total_ops" not in key and "total_params" not in key:
        #         result_dict[result_key] = weight
        
        model.load_state_dict(state_dict)
        # model.load_state_dict(result_dict)  # 验收添加
        logger.info("pretrained model loaded from {}".format(args.pretrained))

    # Model flops and params counting
    image = torch.randn([1, 2, 32, 32])
    flops, params = thop.profile(model, inputs=(image,), verbose=False)
    flops, params = thop.clever_format([flops, params], "%.3f")

    # Model info logging
    logger.info(f'=> Model Name: CRNet [pretrained: {args.pretrained}]')
    logger.info(f'=> Model Config: compression ratio=1/{args.cr}')
    logger.info(f'=> Model Flops: {flops}')
    logger.info(f'=> Model Params Num: {params}\n')
    logger.info(f'{line_seg}\n{model}\n{line_seg}\n')

    return model


def init_my_model(args):
    # Model loading
    model = mynet(args)

    # Model flops and params counting
    image = torch.randn([1, 2, 32, 32])
    flops, params = thop.profile(model, inputs=(image,), verbose=False)
    flops, params = thop.clever_format([flops, params], "%.3f")

    # Model info logging
    logger.info(f'=> Model Name: Entroformer [pretrained: {args.pretrained}]')
    logger.info(f'=> Model Config: compression ratio=1/{args.cr}')
    logger.info(f'=> Model Flops: {flops}')
    logger.info(f'=> Model Params Num: {params}\n')
    logger.info(f'{line_seg}\n{model}\n{line_seg}\n')

    return model
