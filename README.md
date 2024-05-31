# AI System上手任务2

## 任务描述

使用PyTorch和transformers API从0构建一个llama微调脚本，需要支持gradient checkpoint、混精度、数据并行、张量并行4个可选参数，不使用ColossalAI/Megatron/DeepSpeed，可以参考代码


## 训练模型

torchrun --nproc_per_node 4 sft.py --tp --dp --amp --gckp

## 训练结果

超参数设置：bs=2

| 指标      |    原始模型 | +张量并行  |+gradient checkpoint  |+混精度  |+数据并行  |
| :-------- | --------:| :--: |:--: |:--: |:--: |
| 显存占用  |     66508M  | 47750M*2GPU  | 37692M *2GPU|      45908M*2GPU   |  58658M*4GPU  |
| 训练速度     |   1.08it/s |1.49it/s| 1.13it/s |   3.3it/s |  2.69it/s  |



## 实现方法

Llama使用Transformers提供的模型及pretrained weight

gradient checkpoint使用Transformers自带的gradient checkpoint

混精度使用Pytorch自带的torch.cuda.amp

数据并行使用Pytorch自带的DDP

张量并行参考了Megatron的实现

## TBD

张量并行内部的dropout层使用不同的random seed

将lm_head 和CrossEntropy也张量并行化
