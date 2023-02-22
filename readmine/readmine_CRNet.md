填入路径sys.path.append、config类中

error:
main.py引用路径错误，trainer/tester在solver.py中,init_device/init_model在init.py，FakeLR/WarmUpCosineAnnealingLR在scheduler.py
修改solver.py第6，7行import：CRNet_backup->CRNet
注释去config覆盖args，修改为config用args赋值，删去--data_dir参数
修改args参数“-”连接为“_”连接

需在csi_dataset中增加testloader方法，在main中调用获得loader，写入train.loop
testloader同样需要归一化
想法：可用softmax进行归一化