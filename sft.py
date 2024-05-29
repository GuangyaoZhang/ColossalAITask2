
from datasets import load_dataset
import torch.distributed as dist
import random
import numpy as np
import torch
from torch import nn
import os

from llama_tp import TP_LlamaForCausalLM, TP_cross_entropy
def seed_everything(seed=11):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_dataloader():
    from transformers import AutoTokenizer

    dataset = load_dataset('wikitext', 'wikitext-103-v1')

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

    # tokenizer.model_max_length=1024
    tokenizer.pad_token = tokenizer.eos_token


    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True,max_length=512)


    _ = tokenizer("Dummy text", truncation=True)

    tokenized_datasets = dataset.map(tokenize_function, batched=True, load_from_cache_file=True,batch_size=10000,num_proc=50)

    tokenized_datasets = tokenized_datasets.remove_columns(["text"])
    tokenized_datasets.set_format("torch")

    small_train_dataset = tokenized_datasets["train"].shuffle(seed=22).select(range(1000))

    small_eval_dataset = tokenized_datasets["test"].shuffle(seed=22).select(range(1000))
    from torch.utils.data import DataLoader

    train_dataloader = DataLoader(small_train_dataset, shuffle=False, batch_size=1)
    eval_dataloader = DataLoader(small_eval_dataset, batch_size=2)

    return train_dataloader, eval_dataloader

# Copy from Megatron
class _VocabParallelCrossEntropy(torch.autograd.Function):
    """
    分布式计算Loss    
    """
    @staticmethod
    def forward(ctx, vocab_parallel_logits, target):
        # 1. logit - global max(logit)操作，主要目的是防溢出
        logits_max = torch.max(vocab_parallel_logits, dim=-1)[0] # (b, s, 1)
        torch.distributed.all_reduce( # (b, s, 1)
            logits_max,
            op=torch.distributed.ReduceOp.MAX, # 找全局最大值
        )
        # Subtract the maximum value. 
        vocab_parallel_logits.sub_(logits_max.unsqueeze(dim=-1)) # 原始GPU上维护的logits减去每行最大值（防止溢出）

    
        # 2、根据当前进程id，取出当前进程所维护词表序号等信息
        # 函数，能够获取当前进程所维护词表的start_index和end_index
        # 这块GPU上logits最后一维的大小，等于所维护的词表的大小（v/N）
        partition_vocab_size = vocab_parallel_logits.size()[-1]
        # 取得当前进程所在TP组中的序号
        rank = dist.get_rank()
        # 取得当前进程所在TP组的总进程数
        world_size = 2
        # 取得当前进程所维护的词表的start_index和end_index 
        vocab_start_index, vocab_end_index = (0, 16000) if rank==0 else (16000, 32000)

        # 3. 基于真值，取出每个token在真值位置上的logit（即和真值的相似度）
        # Create a mask of valid vocab ids (1 means it needs to be masked)
        target_mask = (target < vocab_start_index) | (target >= vocab_end_index) # target = (b, s)
        masked_target = target.clone() - vocab_start_index
        masked_target[target_mask] = 0

        # Get predicted-logits = logits[target].
        # For Simplicity, we convert logits to a 2-D tensor with size
        # [*, partition-vocab-size] and target to a 1-D tensor of size [*].
        logits_2d = vocab_parallel_logits.view(-1, partition_vocab_size) # (b*s, v/N)
        masked_target_1d = masked_target.view(-1) # (b*s)
        arange_1d = torch.arange( # [b*s]
            start=0, end=logits_2d.size()[0], device=logits_2d.device
        )
        # logits_2d[arange_1d, masked_target_1d]: 
        # tensor的切片操作。arange_1d：取出所有的行。masked_target_1d：取出logit
        predicted_logits_1d = logits_2d[arange_1d, masked_target_1d] # (b*s)
        predicted_logits_1d = predicted_logits_1d.clone().contiguous()
        predicted_logits = predicted_logits_1d.view_as(target) # (b, s)
        predicted_logits[target_mask] = 0.0
        # All reduce is needed to get the chunks from other GPUs.
        torch.distributed.all_reduce( # allreduce之后得到的logit矩阵为(b, s)，每一个位置表示对应真值位置的预测logit
            predicted_logits,
            op=torch.distributed.ReduceOp.SUM,

        )

        # Sum of exponential of logits along vocab dimension across all GPUs.
        exp_logits = vocab_parallel_logits # （b, s, v/N）
        torch.exp(vocab_parallel_logits, out=exp_logits)
        sum_exp_logits = exp_logits.sum(dim=-1) # (b, s)
        torch.distributed.all_reduce(
            sum_exp_logits,
            op=torch.distributed.ReduceOp.SUM,
        )

        # 4. 计算Loss = log(sum(exp(logits))) - predicted-logit.
        loss = torch.log(sum_exp_logits) - predicted_logits # (b, s)

        # Store softmax, target-mask and masked-target for backward pass.
        exp_logits.div_(sum_exp_logits.unsqueeze(dim=-1))
        ctx.save_for_backward(exp_logits, target_mask, masked_target_1d)

        return loss

    @staticmethod
    def backward(ctx, grad_output):

        # Retreive tensors from the forward path.
        softmax, target_mask, masked_target_1d = ctx.saved_tensors

        # All the inputs have softmax as their gradient.
        grad_input = softmax
        # For simplicity, work with the 2D gradient.
        partition_vocab_size = softmax.size()[-1]
        grad_2d = grad_input.view(-1, partition_vocab_size)

        # Add the gradient from matching classes.
        arange_1d = torch.arange(start=0, end=grad_2d.size()[0], device=grad_2d.device)
        grad_2d[arange_1d, masked_target_1d] -= 1.0 - target_mask.view(-1).float()

        # Finally elementwise multiplication with the output gradients.
        grad_input.mul_(grad_output.unsqueeze(dim=-1))

        return grad_input, None



def train(amp, gckp, tp):

    from tqdm import tqdm


    from torch.optim import SGD
    model = TP_LlamaForCausalLM.from_pretrained_tp("meta-llama/Llama-2-7b-hf", tp=tp)

    if gckp:
        model.gradient_checkpointing_enable()
    model.train()
    device = torch.device("cuda")
    model.to(device)
    optimizer = SGD(model.parameters(), lr=5e-5)

    train_dataloader, eval_dataloader = get_dataloader()

    for epoch in range(10):
        sumed_loss = 0
        for batch in tqdm(train_dataloader):
            
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch['input_ids'].clone()

            with torch.cuda.amp.autocast(enabled=amp):
                outputs = model(**batch,return_dict=False)
            
                if False:
                    loss_func = _VocabParallelCrossEntropy
                    loss = loss_func.apply(outputs[0],labels)
                else:
                    loss_func = nn.CrossEntropyLoss()
                    loss = loss_func(outputs[0].view(-1, outputs[0].size(-1)),labels.view(-1))

            sumed_loss+=loss.item()

            loss.backward()

            optimizer.step()

            optimizer.zero_grad()
        print(sumed_loss)


def setup_dist(tp, dp):

    dist.init_process_group("nccl")

    rank = dist.get_rank()
    torch.cuda.set_device(rank)

import argparse


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--tp', action="store_true")
    parser.add_argument('--dp', action="store_true")
    parser.add_argument('--amp', action="store_true")
    parser.add_argument('--gckp', action="store_true")
    
    args = parser.parse_args()
    
    seed_everything()
    setup_dist(args.tp, args.dp)

    train(args.amp, args.gckp, args.tp)

    
