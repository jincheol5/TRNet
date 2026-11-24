import torch
import torch.nn as nn
import torch.nn.functional as F
from .mlp_module import MLP

class Info(nn.Module):
    def __init__(self,
            n_node:int,
            info_dim:int,
            seed:int
        ):
        super().__init__()
        self.n_node=n_node
        self.info_dim=info_dim

        # src information vector: Random Gaussian 벡터 + L2 정규화
        generator=torch.Generator().manual_seed(seed)
        src_info=torch.randn(n_node+1,info_dim,generator=generator)
        src_info=F.normalize(src_info,p=2,dim=-1)
        self.register_buffer("src_info",src_info)

        # dst information vector (cumulative vector)
        self.register_buffer("dst_info",src_info.clone())

        # previous dst information vector
        self.register_buffer("pre_dst_info",src_info.clone())

        # last update time of dst_info, 마지막으로 정보를 수신한 시간.
        self.register_buffer("last_t",torch.full((n_node+1,),-1,dtype=torch.long))

    def init_info(self):
        self.dst_info.copy_(self.src_info)
        self.pre_dst_info.copy_(self.src_info)
        self.last_t.fill_(-1)

    def detach_info(self):
        self.src_info.detach_()
        self.dst_info.detach_()
        self.pre_dst_info.detach_()
        self.last_t.detach_()

class Combiner(nn.Module):
    """
    Gated Combine
    """
    def __init__(self,
            info_dim:int,
            latent_dim:int    
        ):
        super().__init__()
        self.combine_mlp=MLP(
            input_dim=2*info_dim,
            latent_dim=latent_dim,
            output_dim=info_dim
        )
        self.gate=nn.Linear(2*info_dim,info_dim)
        self.info_mlp=MLP(
            input_dim=2*info_dim,
            latent_dim=latent_dim,
            output_dim=info_dim
        )

    def forward(self,
            cur_info:torch.Tensor,
            new_info:torch.Tensor
        ):
        concat_info=torch.cat([cur_info,new_info],dim=-1)
        combined_info=self.combine_mlp(concat_info)
        gate=torch.sigmoid(self.gate(concat_info))
        gated_info=gate*combined_info
        final_info=torch.cat([cur_info,gated_info])
        return self.info_mlp(final_info)
