# 请先阅读README.md文件
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
sys.path.append("../../")                  # 将/path/to/your/code/改为实际的代码工作目录
import torch
import torch.nn as nn
from csi_reference.CRNet.utils.parser import args
from csi_reference.CRNet.utils import logger
from csi_reference.CRNet.utils.solver import Trainer, Tester
from csi_reference.CRNet.utils.init import init_device, init_model
from csi_reference.CRNet.utils.scheduler import FakeLR, WarmUpCosineAnnealingLR
from csi_reference.modelSave_stack import ModelSave_stack
from csi_reference.CRNet.dataset.csi_datasets import get_loader

class config:
    csi_dims = (2, 32, 32)
    trainset = args.scenario        # 场景，可选项为“indoor"/”outdoor“
    base_path = '../../csi_reference/CRNet/'
    data_dir = '../../csi_reference/CRNet/dataset/%sdoor/' % trainset      # 需要将数据集置入到CRNet/dataset/目录下
    batch_size = args.batch_size
    epochs = args.epochs
    cr = args.cr      # 压缩率，可选项为4、8、16、32、64，对应输出维度为512、256、128、64、32，每个CR对应一个模型结构
    pretrained = "../../csi_reference/CRNet/checkpoints/%s_%02d.pth" % (trainset, cr)
    workdir = "../../csi_reference/CRNet/history/"  # model save
    if not os.path.exists(workdir):
        os.makedirs(workdir)
    # epochLog = workdir + "/epochLog_cr%d.csv" % cr
    epochLog = workdir + "/epochLog_softmax.csv"


# args.batch_size = config.batch_size
# args.epochs = config.epochs
# args.cr = config.cr
if not args.evaluate:
    args.pretrained = config.pretrained  # mdf该行代码开启加载预训练模型
    modelSave_stack = ModelSave_stack(config, stack_size=1)  # 只需保留最优 原stack_size=5
    epochLogger = open(config.epochLog, "w")


def main():
    logger.info('=> PyTorch Version: {}'.format(torch.__version__))

    # Environment initialization
    device, pin_memory = init_device(args.seed, args.cpu, args.gpu, args.cpu_affinity)

    # Create the data loader
    train_loader, val_loader = get_loader(config)
    # train_loader, val_loader, test_loader = Cost2100DataLoader(
    #     root=args.data_dir,
    #     batch_size=args.batch_size,
    #     num_workers=args.workers,
    #     pin_memory=pin_memory,
    #     scenario=args.scenario)()

    # Define model
    model = init_model(args)  # mdf在此修改模型
    model.to(device)

    # Define loss function
    criterion = nn.MSELoss().to(device)

    # Inference mode-------------------------------------------------------------------------------------
    if args.evaluate:
        Tester(model, device, criterion)(val_loader)
        return
    # ---------------------------------------------------------------------------------------------------
    # Define optimizer and scheduler
    lr_init = 1e-3 if args.scheduler == 'const' else 2e-3
    optimizer = torch.optim.Adam(model.parameters(), lr_init)
    if args.scheduler == 'const':
        scheduler = FakeLR(optimizer=optimizer)
    else:
        scheduler = WarmUpCosineAnnealingLR(optimizer=optimizer,
                                            T_max=args.epochs * len(train_loader),
                                            T_warmup=30 * len(train_loader),
                                            eta_min=5e-5)

    # Define the training pipeline
    trainer = Trainer(model=model,
                      device=device,
                      optimizer=optimizer,
                      criterion=criterion,
                      scheduler=scheduler,
                      resume=args.resume,
                      save_path=config.workdir,
                      modelSaver=modelSave_stack,
                      logger=epochLogger)

    # Start training
    trainer.loop(args.epochs, train_loader, val_loader, test_loader=val_loader)

    # Final testing
    loss, nmse = Tester(model, device, criterion)(val_loader)
    print(f"\n=! Final test loss: {loss:.3e}"
          f"\n         test NMSE: {nmse:.3e}\n")


if __name__ == "__main__":
    main()