import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from graph import CAW_N_Graph
from module import MLP,TimeEncoder

class CAW_N(nn.Module):
    def __init__(self,
            node_dim:int,
            edge_dim:int,
            latent_dim:int,
            time_dim:int,
            embed_dim:int,
            n_head:int,
            n_walk:int,
            walk_len:int
        ):
        super().__init__()
        self.node_dim=node_dim
        self.edge_dim=edge_dim
        self.latent_dim=latent_dim
        self.time_dim=time_dim
        self.embed_dim=embed_dim
        self.n_head=n_head
        self.n_walk=n_walk
        self.walk_len=walk_len

        # time encoder
        self.time_encoder=TimeEncoder(time_dim=time_dim)

        # anonymized_vec encoder
        self.anonymized_vec_encoder=MLP(
            input_dim=walk_len+1,
            latent_dim=latent_dim,
            output_dim=node_dim
        )

        # walk sequence encoder
        self.rnn_encoder=nn.GRU(
            input_size=node_dim+time_dim+node_dim+edge_dim,
            hidden_size=embed_dim,
            num_layers=1,
            batch_first=True
        )

        # attn module for walks aggregation (CAW-N 공식 Github)
        self.walk_attn=nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_head,
            dim_feedforward=4*embed_dim,
            dropout=0.1,
            activation="relu",
            batch_first=True
        )

        # mean pooling & linear for walks aggregation (CAW-N 공식 Github)
        self.walk_pooler=nn.Sequential(
            nn.Linear(embed_dim,embed_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # decoder
        self.decoder=MLP(
            input_dim=2*embed_dim,
            latent_dim=latent_dim,
            output_dim=1
        )

    def set_graph(self,graph:CAW_N_Graph):
        self.graph=graph
        self.n_node=graph.n_node

    def embedding(self,
            src:torch.Tensor,
            dst:torch.Tensor,
            event_t:torch.Tensor
        ):
        """
        S_u, S_v에서 각각 M개씩 walk embedding 생성.
        """
        ### 1. Set Based Anonymization
        anonymization_result=self.graph.set_based_anonymization(
            src=src,
            dst=dst,
            event_t=event_t
        )

        # load to GPU or MPS
        device=next(self.parameters()).device
        anonymization_result={
            name:tensor.to(device)
            for name,tensor in anonymization_result.items()
        }

        src_anonymized_walks=anonymization_result["src_anonymized_walks"] # [B,M,m+1,2,m+1]
        src_walks_node=anonymization_result["src_walks_node"] # [B,M,m+1]
        src_walks_t=anonymization_result["src_walks_t"] # [B,M,m+1]
        src_walks_edge=anonymization_result["src_walks_edge"] # [B,M,m+1]
        src_walks_mask=anonymization_result["src_walks_mask"] # [B,M,m+1]

        dst_anonymized_walks=anonymization_result["dst_anonymized_walks"] # [B,M,m+1,2,m+1]
        dst_walks_node=anonymization_result["dst_walks_node"] # [B,M,m+1]
        dst_walks_t=anonymization_result["dst_walks_t"] # [B,M,m+1]
        dst_walks_edge=anonymization_result["dst_walks_edge"] # [B,M,m+1]
        dst_walks_mask=anonymization_result["dst_walks_mask"] # [B,M,m+1]

        ### 2-1. encode S_u
        # encode src_anonymized_walks
        src_encoded_walks=self.anonymized_vec_encoder(
            src_anonymized_walks.to(dtype=torch.float32)
        ) # [B,M,m+1,2,node_dim]
        src_encoded_walks=src_encoded_walks.sum(dim=-2) # [B,M,m+1,node_dim]

        # encode src_walks_t
        src_delta_t=torch.cat([
            torch.zeros_like(src_walks_t[...,:1]),
            src_walks_t[...,:-1]-src_walks_t[...,1:]
        ],dim=-1) # [B,M,m+1]
        src_delta_t=src_delta_t.unsqueeze(-1) # [B,M,m+1,1]
        src_delta_t_vec=self.time_encoder(src_delta_t) # [B,M,m+1,time_dim]

        # apply RNN with mask
        src_walks_node_vec=self.graph.node_ft[src_walks_node] # [B,M,m+1,node_dim]
        src_walks_edge_vec=self.graph.edge_ft[src_walks_edge] # [B,M,m+1,edge_dim]
        src_walks_vec=torch.cat([
            src_encoded_walks,
            src_delta_t_vec,
            src_walks_node_vec,
            src_walks_edge_vec
        ],dim=-1) # [B,M,m+1,node_dim+time_dim+node_dim+edge_dim]
        B,M,L,D=src_walks_vec.shape
        walk_input=src_walks_vec.reshape(B*M,L,D)
        walk_lengths=src_walks_mask.reshape(B*M,L).long().sum(dim=-1) # [B*M],실제 Walk 길이 (Root Node 포함)
        packed_input=pack_padded_sequence(
            walk_input,
            lengths=walk_lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )
        _,h_n=self.rnn_encoder(packed_input)
        src_walks_embed=h_n[-1].reshape(B,M,-1) # [B,M,embed_dim]

        ### 2-2. encode S_v
        # encode dst_anonymized_walks
        dst_encoded_walks=self.anonymized_vec_encoder(
            dst_anonymized_walks.to(dtype=torch.float32)
        ) # [B,M,m+1,2,node_dim]
        dst_encoded_walks=dst_encoded_walks.sum(dim=-2) # [B,M,m+1,node_dim]

        # encode src_walks_t
        dst_delta_t=torch.cat([
            torch.zeros_like(dst_walks_t[...,:1]),
            dst_walks_t[...,:-1]-dst_walks_t[...,1:]
        ],dim=-1) # [B,M,m+1]
        dst_delta_t=dst_delta_t.unsqueeze(-1) # [B,M,m+1,1]
        dst_delta_t_vec=self.time_encoder(dst_delta_t) # [B,M,m+1,time_dim]

        # apply RNN with mask
        dst_walks_node_vec=self.graph.node_ft[dst_walks_node] # [B,M,m+1,node_dim]
        dst_walks_edge_vec=self.graph.edge_ft[dst_walks_edge] # [B,M,m+1,edge_dim]
        dst_walks_vec=torch.cat([
            dst_encoded_walks,
            dst_delta_t_vec,
            dst_walks_node_vec,
            dst_walks_edge_vec
        ],dim=-1) # [B,M,m+1,node_dim+time_dim+node_dim+edge_dim]
        B,M,L,D=dst_walks_vec.shape
        walk_input=dst_walks_vec.reshape(B*M,L,D)
        walk_lengths=dst_walks_mask.reshape(B*M,L).long().sum(dim=-1) # [B*M],실제 Walk 길이 (Root Node 포함)
        packed_input=pack_padded_sequence(
            walk_input,
            lengths=walk_lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )
        _,h_n=self.rnn_encoder(packed_input)
        dst_walks_embed=h_n[-1].reshape(B,M,-1) # [B,M,embed_dim]

        ### 3. walks aggregation
        # src walk aggregation
        src_attn=self.walk_attn(src_walks_embed) # [B,M,embed_dim]
        src_embed=self.walk_pooler(src_attn.mean(dim=1)) # [B,embed_dim]

        # dst walk aggregation
        dst_attn=self.walk_attn(dst_walks_embed) # [B,M,embed_dim]
        dst_embed=self.walk_pooler(dst_attn.mean(dim=1)) # [B,embed_dim]
        return {
            "src_vec":src_embed, # [B,embed_dim]
            "dst_vec":dst_embed # [B,embed_dim]
        }

    def forward(self,
            src:torch.Tensor,
            dst:torch.Tensor,
            event_t:torch.Tensor,
        ):
        """
        """
        ### embedding
        embedded_vec=self.embedding(
            src=src,
            dst=dst,
            event_t=event_t
        )
        src_vec=embedded_vec["src_vec"] # [B,embed_dim]
        dst_vec=embedded_vec["dst_vec"] # [B,embed_dim]

        ### decode 
        pair_vec=torch.concat([src_vec,dst_vec],dim=-1)
        pred_logit=self.decoder(pair_vec) # [B,1]
        return pred_logit.squeeze(-1) # [B,]