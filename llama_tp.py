from typing import Any
from transformers import LlamaForCausalLM
from transformers.models.llama.modeling_llama import \
      LlamaModel,LlamaMLP,LlamaRMSNorm,LlamaAttention, \
        LlamaDecoderLayer,LlamaForCausalLM,LlamaSdpaAttention

import torch

import torch.distributed as dist
from torch import nn
from torch.autograd import Function



tp_group=[]

class Embedding_TP(nn.Module):

    def __init__(self, embs):
        super().__init__()
        embs:nn.Embedding
        rank = dist.get_rank()
        rank = dist.get_group_rank(tp_group, rank)
        vocab_size = len(embs.weight)
        per_rank_vocab = vocab_size//2

        self.per_rank_vocab = per_rank_vocab

        emb_weight_shard = embs.weight[per_rank_vocab*rank:per_rank_vocab*(rank+1)]
        self.embedding = nn.Embedding(per_rank_vocab,embs.weight.size(1), _weight=emb_weight_shard)
    @staticmethod
    def from_full_model(full_model):
        embedding = full_model
        return Embedding_TP(embedding)
    
    def forward(self,input):
        input = input.clone()
        # rank = dist.get_rank()
        # rank = dist.get_group_rank(tp_group, rank)
        rank = 0
        input-=rank*self.per_rank_vocab
        mask = (input<self.per_rank_vocab )*(input>=0 )
        
        input[~mask]=2
        
        emb = self.embedding(input)

        emb[~mask] = 0

        dist.all_reduce(torch.rand(10000).cuda(),group=tp_group)
        dist.barrier()

    
        # dist.all_reduce(torch.zeros_like(emb),group=tp_group)
        # dist.barrier()

        dist.all_reduce(torch.ones_like(emb),group=tp_group)
        dist.barrier()

        dist.all_reduce(emb.clone(),group=tp_group)
        dist.barrier()

        dist.all_reduce(emb,group=tp_group)
        # dist.all_reduce(emb)

        dist.barrier()
        # import time
        # time.sleep(1)
        # exit(0)




        return emb
        
    
class F(Function):
    @staticmethod
    def forward(ctx, x):
        return x

    @staticmethod
    def backward(ctx, gradients) :
        dist.all_reduce(gradients,group=tp_group)
        return gradients


class TP_Linear1(nn.Module):
    def __init__(self, linear) -> None:
        super().__init__()
        linear:nn.Linear
        linear.weight

        self.per_rank_size = linear.weight.size(0)//2


        self.linear = nn.Linear(linear.weight.size(1),self.per_rank_size,bias=linear.bias)
        rank = dist.get_rank()
        rank = dist.get_group_rank(tp_group, rank)
        self.linear.weight = nn.Parameter(linear.weight[rank*self.per_rank_size:(rank+1)*self.per_rank_size])
        if linear.bias:
            self.linear.bias = nn.Parameter(linear.bias[rank*self.per_rank_size:(rank+1)*self.per_rank_size])
    @staticmethod
    def from_full_model(full_model):
        return TP_Linear1(full_model)
    def forward(self, input):
        input = F.apply(input)
        return self.linear(input)



class TP_Linear2(nn.Module):
    def __init__(self, linear) -> None:
        super().__init__()


        self.per_rank_size = linear.weight.size(1)//2
        self.linear = nn.Linear(self.per_rank_size,linear.weight.size(0),bias=linear.bias)
        
        rank = dist.get_rank()
        rank = dist.get_group_rank(tp_group, rank)
        self.linear.weight = nn.Parameter(linear.weight[:, rank*self.per_rank_size:(rank+1)*self.per_rank_size])
        
        if linear.bias:
            self.linear.bias = nn.Parameter(linear.bias/2)


    @staticmethod
    def from_full_model(full_model):
        return TP_Linear2(full_model)

    def forward(self, input):
        output = self.linear(input)
        dist.all_reduce(output,group=tp_group)
        return output



class TP_LlamaMLP(LlamaMLP):
    @staticmethod
    def from_full_model(full_model):
        full_model:LlamaMLP
        full_model.gate_proj = TP_Linear1.from_full_model(full_model.gate_proj)
        full_model.up_proj = TP_Linear1.from_full_model(full_model.up_proj)

        full_model.down_proj = TP_Linear2.from_full_model(full_model.down_proj)
        

        return full_model


class TP_LlamaDecoderLayer(LlamaDecoderLayer):
    @staticmethod
    def from_full_model(full_model):
        full_model:LlamaDecoderLayer
        full_model.self_attn = full_model.self_attn
        full_model.mlp = TP_LlamaMLP.from_full_model(full_model.mlp)
        return full_model


class TP_LlamaAttention(LlamaSdpaAttention):
    @staticmethod
    def from_full_model(full_model):
        full_model:LlamaSdpaAttention
        full_model.q_proj = TP_Linear1(full_model.q_proj)
        full_model.k_proj = TP_Linear1(full_model.k_proj)
        full_model.v_proj = TP_Linear1(full_model.v_proj)
        full_model.num_heads = full_model.num_heads//2
        full_model.hidden_size = full_model.hidden_size//2
        
        full_model.o_proj = TP_Linear2(full_model.o_proj)


        return full_model

class TP_LlamaModel(LlamaModel):
    @staticmethod
    def from_full_model(full_model):
        full_model:LlamaModel
        full_model.embed_tokens = Embedding_TP.from_full_model(full_model.embed_tokens)
        full_model.layers = nn.ModuleList(
            [TP_LlamaDecoderLayer.from_full_model(layer) for layer in full_model.layers]
        )
        return full_model

class TP_LlamaForCausalLM(LlamaForCausalLM):

    @staticmethod
    def from_pretrained_tp(model_name, tpg):
        model = TP_LlamaForCausalLM.from_pretrained(model_name)
        # return model 
        if tpg is None:
            return model 
        else:
            print("TP")
            global tp_group
            tp_group = tpg
            model.model = TP_LlamaModel.from_full_model(model.model)
            # model.lm_head = TP_Linear1.from_full_model(model.lm_head)


        return model



# class TP_cross_entropy(nn.Module):
#     def __init__(self):
       
#         super().__init__()

#         self.local_cross_entropy = nn.CrossEntropyLoss()


#     def forward(self,logits, labels):
#         shift_logits = logits[..., :-1, :].contiguous()
#         shift_labels = labels[..., 1:].contiguous()
        
#         logits_exps = torch.exp(shift_logits)
#         logits_exp_sum_per_rank = torch.sum(logits_exps,dim=-1)
#         logits_exp_sum_per_rank = logits_exp_sum_per_rank.view(-1)

#         dist.all_reduce(logits_exp_sum_per_rank)

#         # logits_exp_sum_per_rank = torch.unsqueeze(logits_exp_sum_per_rank, -1)

#         # logits_exps = logits_exps/logits_exp_sum_per_rank


#         shift_labels = shift_labels-shift_logits.size(2)*dist.get_rank()

#         label_mask = (shift_labels>=0)*(shift_labels<shift_logits.size(2))
#         shift_labels[~label_mask]=0


#         label_mask = label_mask.view(-1)
#         shift_labels = shift_labels.view(-1)
#         shift_logits = shift_logits.view(-1, shift_logits.size(2))

#         logits_arange = torch.arange(0, len(shift_labels))

#         true_logits = shift_logits[logits_arange, shift_labels]
#         true_logits[~label_mask] = 0
        


#         dist.all_reduce(true_logits)


#         return torch.mean(torch.log(logits_exp_sum_per_rank)-true_logits)

