import pandas as pd
import numpy as np
import torch
from .GNN_graph import GNN_Graph

class DyGFormer_Graph(GNN_Graph):
    def __init__(self,graph_df:pd.DataFrame):
        super().__init__(graph_df)

    def get_history_seq(self,
            node:torch.Tensor,
            event_t:torch.Tensor,
            max_history_len:int
        )->dict[str,list[np.ndarray]]:
        """
        배치 내 각 노드의 과거 1-hop 이력을 시퀀스로 반환한다.
        자기 자신을 맨 앞에 넣고, 남은 길이만큼 가장 최근 이력들을 가져와서 그 중에서 오래된 순서부터 붙인다. 
        자기 자신의 edge ID와 시간 차이는 0이다.
        """
        max_history_len=max_history_len-1
        node_seq_list=[]
        edge_seq_list=[]
        ts_seq_list=[]

        for node_id,cut_time in zip(node.cpu().numpy(),event_t.cpu().numpy()):
            node_id=int(node_id)
            cut_time=int(cut_time)

            times=self.in_adj_t[node_id]
            cut_idx=int(np.searchsorted(times,cut_time,side="right")) # 같은 시각의 이력 포함
            start_idx=max(0,cut_idx-max_history_len)

            selected_neighbors=self.in_adj[node_id][start_idx:cut_idx]
            selected_edges=self.in_adj_edge[node_id][start_idx:cut_idx]
            selected_times=np.asarray(times[start_idx:cut_idx])

            node_seq_list.append(np.asarray([node_id,*selected_neighbors],dtype=np.int64))
            edge_seq_list.append(np.asarray([0,*selected_edges],dtype=np.int64))
            ts_seq_list.append(np.concatenate([
                np.asarray([0],dtype=np.int64),
                (cut_time-selected_times).astype(np.int64)
            ]))
        return {
            "node":node_seq_list,
            "edge":edge_seq_list,
            "ts":ts_seq_list
        }
