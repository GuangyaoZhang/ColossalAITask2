# AI System上手任务2

## 任务描述

使用PyTorch和transformers API从0构建一个llama微调脚本，需要支持gradient checkpoint、混精度、数据并行、张量并行4个可选参数，不使用ColossalAI/Megatron/DeepSpeed，可以参考代码



## 训练模型

torchrun --nproc_per_node 2 sft.py --tp --amp --gckp

## 训练结果

超参数设置：bs=2

| 指标      |    原始模型 | +张量并行  |+gradient checkpoint  |+混精度  |+数据并行  |
| :-------- | --------:| :--: |:--: |:--: |:--: |
| 显存占用  |     59524M  | 42486M  | 36830M|      44928M   |    |
| 训练速度     |   2.03it/s |2.78it/s| 2.17it/s |    6.58it/s |    |



## 实现方法

Llama使用Transformers提供的模型及pretrained weight
gradient checkpoint使用Transformers自带的gradient checkpoint
混精度使用Pytorch自带的torch.cuda.amp
数据并行使用Pytorch自带的ddp
张量并行参考了Megatron的实现
