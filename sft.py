
from datasets import load_dataset
import torch.distributed as dist
import random
import numpy as np
import torch
from torch import nn
import os
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
import argparse

from llama_tp import TP_LlamaForCausalLM

def seed_everything(seed=11):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_dataloader(dp):
    if dp:
        local_rank = dist.get_group_rank(dp, dist.get_rank())
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

    if dp:
        if local_rank==0:
            idx = range(0,50)
        else:
            idx = range(50,100)
    else:
        idx = range(0,100)
    small_train_dataset = tokenized_datasets["train"].shuffle(seed=22).select(idx)

    small_eval_dataset = tokenized_datasets["test"].shuffle(seed=22).select(idx)
    from torch.utils.data import DataLoader

    train_dataloader = DataLoader(small_train_dataset, shuffle=False, batch_size=2)
    eval_dataloader = DataLoader(small_eval_dataset, batch_size=2)

    return train_dataloader, eval_dataloader


def train(amp, gckp, tp_group, dp_group):

    from tqdm import tqdm

    from torch.optim import SGD
    model = TP_LlamaForCausalLM.from_pretrained_tp("meta-llama/Llama-2-7b-hf", tp_group)

    if gckp:
        model.gradient_checkpointing_enable()
    
    model.train()
    device = torch.device("cuda")
    model.to(device)
    if dp_group:
        dist.barrier()
        model = DDP(model, process_group=dp_group)
        dist.barrier()


    optimizer = SGD(model.parameters(), lr=1e-4 if dp_group else 5e-5)

    train_dataloader, eval_dataloader = get_dataloader(dp_group)

    if amp:
        scaler = torch.cuda.amp.GradScaler()
    sumed_losses = []
    for epoch in range(10):
        sumed_loss = 0

        for batch in tqdm(train_dataloader):
            
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch['input_ids'].clone()

            with torch.cuda.amp.autocast(enabled=amp):
                outputs = model(**batch,return_dict=False)
        
                loss_func = nn.CrossEntropyLoss()
                loss = loss_func(outputs[0].view(-1, outputs[0].size(-1)),labels.view(-1))
            sumed_loss+=loss.item()
            if amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            optimizer.zero_grad()
        
        sumed_losses.append(sumed_loss)
    if dist.get_rank()==0:
        print(sumed_losses)


def setup_dist(tp, dp):
    

    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    rank = dist.get_rank()
    torch.cuda.set_device(rank)

    world_size = dist.get_world_size()

    dp_size=2 if dp else 1
    tp_size=2 if tp else 1
    assert world_size == tp_size*dp_size

    #       TP
    #  DP   0 1      
    #       2 3
    if tp:
        if dp:
            dp_groups = [[0,1],[2,3]]
            tp_groups = [[0,2],[1,3]]
            
            dp_groups = [dist.new_group(dp_group) for dp_group in dp_groups]
            tp_groups = [dist.new_group(tp_group) for tp_group in tp_groups]

            dp_group = dp_groups[0] if rank in [0, 1] else dp_groups[1]
            tp_group = tp_groups[0] if rank in [0, 2] else tp_groups[1]
        else:
            dp_group = None
            tp_group = [0, 1]
            tp_group = dist.new_group(tp_group)
    else:
        if dp:
            dp_group = [0, 1]
            dp_group = dist.new_group(dp_group)
            tp_group = None
        else:
            dp_group = None
            tp_group = None

    return tp_group, dp_group

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--tp', action="store_true")
    parser.add_argument('--dp', action="store_true")
    parser.add_argument('--amp', action="store_true")
    parser.add_argument('--gckp', action="store_true")
    

    args = parser.parse_args()
    
    seed_everything()
    tp_group, dp_group = setup_dist(args.tp, args.dp)

    train(args.amp, args.gckp, tp_group, dp_group)