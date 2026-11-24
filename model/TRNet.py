import torch
import torch.nn as nn
from graph import GNN_Graph
from module import MLP,Info,Combiner

class TRNet(nn.Module):
    def __init__(self,
            info_dim:int,
            latent_dim:int
        ):
        super().__init__()
        self.info_dim=info_dim
        self.latent_dim=latent_dim

        # info combiner
        self.combiner=Combiner(
            info_dim=info_dim,
            latent_dim=latent_dim
        )

        # decoder
        self.decoder=MLP(
            input_dim=2*info_dim,
            latent_dim=latent_dim,
            output_dim=1
        )

    def set_graph(self,graph:GNN_Graph):
        self.graph=graph
        self.n_node=graph.n_node
        self.info=Info(
            n_node=self.n_node,
            info_dim=self.info_dim,
            seed=torch.initial_seed()
        )

    def update_info(self,
            src:torch.Tensor,
            dst:torch.Tensor,
            event_t:torch.Tensor
        ):
        for u,v,t in zip(src,dst,event_t):
            u_last_t=self.info.last_t[u]
            if u_last_t==t:
                trans_info=self.info.pre_dst_info[u].clone()
            else: # last_t(u) < event_t 
                trans_info=self.info.dst_info[u].clone()
            cur_info=self.info.dst_info[v].clone()
            if self.info.last_t[v]<t:
                self.info.pre_dst_info[v]=cur_info
            self.info.dst_info[v]=self.combiner(cur_info=cur_info,new_info=trans_info)
            self.info.last_t[v]=t

    def forward(self,
            src:torch.Tensor,
            dst:torch.Tensor
        ):
        src_vec=self.info.src_info[src]
        dst_vec=self.info.dst_info[dst]

        ### decode 
        pair_vec=torch.concat([src_vec,dst_vec],dim=-1)
        pred_logit=self.decoder(pair_vec) # [B,1]
        return pred_logit.squeeze(-1) # [B,]