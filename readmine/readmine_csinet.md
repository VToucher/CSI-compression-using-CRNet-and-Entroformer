tensorflow与keras版本、python版本有对应关系，安装tensorflow=2.1.0，keras=2.3.1，python=3.6,cudatoolkit=10.0.130,cudnn=7.6.4
另需安装numpy、matplotlib
添加data文件夹，内放数据集
csinet_train.py第9行改为tf.compat.v1.reset_default_graph()
修改库文件tensorflow_backend第506行devices列表问题https://stackoverflow.com/questions/60581677/experimental-list-devices-attribute-missing-in-tensorflow-core-api-v2-config
修改库文件E:\TOOL\anaconda3\envs\csinet\Lib\site-packages\tensorflow_core\python\keras\callbacks.py第1532行、1732行：self._log_write_dir = self.log_dir
    https://www.jianshu.com/p/9da54361d289
取消计算rho的相关代码csinet_train.py 131-170行
修改file路径名格式，增加timetag区分
修改数据集加载方式与路径
增加nmse输入至文件保存
增加parser.py模块便于参数设置

ERROR:
tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX AVX2
    添加os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
csinet_train.py第105行路径改为\

NMSE为归一化均方误差，通过计算两CSI矩阵各项差的平方和，再用真实值平方和归一
csinet_train.py包含训练与预测，csinet_onlytest用保存的权重直接预测，因为数据集不同，需要重新训练
性能指标采用压缩率CR=1/4、1/8、1/16、1/32、1/64处的NMSE与（rho删去），即训练indoor与outdoor各5个模型，在onlytest中循环预测
    *由于计算rho需要其他数据集支持

下一步训练出所需模型，修改代码在onlytest中循环预测