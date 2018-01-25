# cifarclassify

---
## 图像分类算法
这个仓库主要实现常用的网络并在cifar10数据集上进行试验，比较分类精度。主要参考如下所示：
- [pytorch-cifar](https://github.com/kuangliu/pytorch-cifar)
- [pytorch-classification](https://github.com/bearpaw/pytorch-classification)


---
### 网络实现
- alexnet
- MobileNet [mobilenet实现](doc/mobilenet_implement.md)
- ...

---
### 数据集实现
- cifar10
- ...

---
### 依赖
- pytorch

---
### 用法

**可视化**

[visdom](https://github.com/facebookresearch/visdom)
[网络结构可视化](doc/pytorch_net_visual.md)


```bash
# 在tmux或者另一个终端中开启可视化服务器visdom
python -m visdom.server
# 然后在浏览器中查看127.0.0.1:9097
```

**训练**
```bash
# 训练模型
python train.py
```

**校验**
```bash
# 校验模型
python validate.py
```

**测试**
```bash
# 测试模型
python test.py
```