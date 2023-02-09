import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
sys.path.append("../../")                  # 将/path/to/your/code/改为实际的代码工作目录
import torch
import torch.nn as nn
import numpy as np
from csi_reference.CRNet.utils.parser import args
from csi_reference.CRNet.utils import logger
from csi_reference.CRNet.utils.solver import EntroTrainer, EntroTester
from csi_reference.CRNet.utils.init import init_device, init_my_model
from csi_reference.CRNet.utils.scheduler import LearningRateScheduler
from csi_reference.modelSave_stack import ModelSave_stack
from csi_reference.CRNet.dataset.csi_datasets import get_loader
from my_criterion import *


class config:
    csi_dims = (2, 32, 32)
    trainset = args.scenario        # 场景，可选项为“indoor"/”outdoor“
    base_path = '../../csi_reference/CRNet/'
    data_dir = '../../csi_reference/CRNet/dataset/%sdoor/' % trainset      # 需要将数据集置入到CRNet/dataset/目录下
    batch_size = args.batch_size
    epochs = args.epochs
    cr = args.cr      # 压缩率，可选项为4、8、16、32、64，对应输出维度为512、256、128、64、32，每个CR对应一个模型结构
    # pretrained = "../../csi_reference/CRNet/checkpoints/%s_%02d.pth" % (trainset, cr)
    workdir = "../../csi_reference/CRNet/history/"  # model save
    if not os.path.exists(workdir):
        os.makedirs(workdir)
    epochLog = workdir + "/epochLog_cr%d.csv" % cr
    
    
modelSave_stack = ModelSave_stack(config, stack_size=5)
epochLogger = open(config.epochLog, "w")


def main():
    logger.info('=> PyTorch Version: {}'.format(torch.__version__))

    # Environment initialization
    device, pin_memory = init_device(args.seed, args.cpu, args.gpu, args.cpu_affinity)

    # Create the data loader
    train_loader, val_loader = get_loader(config)
    
    # Define model
    model = init_my_model(args)  # mdf在此修改模型
    model.to(device)
    
    # Define loss function
    criterion_entropy = DiscretizedMixGaussLoss(rgb_scale=False, x_min=-args.table_range, x_max=args.table_range-1,
                                                num_p=args.num_parameter, L=args.table_range*2)
    
    # Inference mode-------------------------------------------------------------------------------------
    if args.evaluate:
        EntroTester(model, device, criterion_entropy, batchSize=args.batch_size, alpha=args.alpha)(val_loader)
        return
    # ---------------------------------------------------------------------------------------------------
    # Define optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, eps=1e-8, weight_decay=0.5)
    lr_step = list(np.linspace(0, args.epochs, 6, dtype=int))[1:]
    lr_scheduler = LearningRateScheduler(mode='stagedecay',
                                            lr=1e-4,
                                            num_training_instances=len(train_loader),
                                            stop_epoch=args.epochs,
                                            warmup_epoch=args.epochs*0.05,
                                            stage_list=lr_step,
                                            stage_decay=0.5)
    lr_scheduler.update_lr(0)


    # Define the training pipeline
    trainer = EntroTrainer(model=model,
                      device=device,
                      optimizer=optimizer,
                      criterion=criterion_entropy,
                      scheduler=lr_scheduler,
                      resume=args.resume,
                      save_path=config.workdir,
                      modelSaver=modelSave_stack,
                      logger=epochLogger,
                      batchSize=args.batch_size,
                      alpha=args.alpha,
                      grad_norm_clip=args.grad_norm_clip)
    
    # Start training
    trainer.loop(args.epochs, train_loader, val_loader, test_loader=val_loader)
    
    # Final testing
    loss, nmse = EntroTester(model, device, criterion_entropy)(val_loader)
    print(f"\n=! Final test loss: {loss:.3e}"
          f"\n         test NMSE: {nmse:.3e}\n")
    
     
    
if __name__ == "__main__":
    main()
