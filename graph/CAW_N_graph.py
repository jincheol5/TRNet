import pandas as pd
import numpy as np
import torch
from .GNN_graph import GNN_Graph

class CAW_N_Graph(GNN_Graph):
    def __init__(self,
            graph_df:pd.DataFrame,
            n_walk:int=10,
            walk_len:int=5
        ):
        super().__init__(graph_df)
        self.n_walk=n_walk # sampling할 walk 개수 = M
        self.walk_len=walk_len # walk의 hop 수 = m

    def get_walks(self,
            root:torch.Tensor,
            root_t:torch.Tensor,
            n_candidate:int=5,
            alpha:float=0.000001
        ):
        """
        Causality Extraction.
        - 특정 link (u,v,t)를 예측할 때, u와 v 각각 m 길이의 backward temporal walk를 M 개 sampling.
        - S_u: node u의 m 길이의 temporal walk M개.
        - S_v: node v의 m 길이의 temporal walk M개.

        Causal Anonymous Walk Sampling.
        - 시간 역순으로 거슬러 올라가며 연결된 link들을 추출.
        - 다음 link 선택은 후보 link들 중 랜덤(alpha=0) or 확률 (alpha>0) 선택.
        - 각 후보 link가 선택될 확률: exp(alpha*(t-t_p))
        - alpha: non-negative hyper-parameter
        - t: 현재 link의 time.
        - t_p: 후보 link의 time.
        - 후보 link가 없는 경우 빈 walk를 0로 padding 처리.

        Efficient Sampling 적용. 
        - node의 과거 link 샘플링 후보를 과거의 전체 이력으로 두지 않고 최근 n_candidate개로 한정한다. 
        - 논문에서는 n_candidate=5 일 때 성능이 가장 좋았음을 주장한다.

        Batch 처리.
        - 각 Root를 M번 복제하여 총 B*M개의 Walk를 생성.
        - 각 Walk의 현재 노드와 시간을 기준으로 과거 링크 후보를 조회.
        - 시간 감쇠 확률에 따라 후보 링크 하나를 샘플링.
        - 선택한 링크의 (node, node_t, edge_id)를 저장.
        - 선택한 node와 node_t를 다음 Hop의 기준으로 사용.
        - 과거 링크가 더 이상 없는 Walk는 종료하고 Padding 처리.
        
        Input:
            root: [B,]
            root_t: [B,]
            n_candidate: int
            alpha: 1e-6
        Return:
            walks_node
            walks_t
            walks_edge
        """
        n_root=root.size(0)

        # Tensor의 각 원소를 M번씩 반복
        cur_node=root.repeat_interleave(self.n_walk) # [B*M,]
        cur_t=root_t.repeat_interleave(self.n_walk) # [B*M,]

        walks_node=torch.zeros(
            (n_root*self.n_walk,self.walk_len+1),
            dtype=torch.long
        ) # [B*M,m+1]
        walks_t=torch.zeros(
            (n_root*self.n_walk,self.walk_len+1),
            dtype=torch.long
        ) # [B*M,m+1]
        walks_edge=torch.zeros(
            (n_root*self.n_walk,self.walk_len+1),
            dtype=torch.long
        ) # [B*M,m+1]

        # Root 정보 저장
        walks_node[:,0]=cur_node
        walks_t[:,0]=cur_t

        active=torch.ones(n_root*self.n_walk,dtype=torch.bool) # [B*M,]
        for hop in range(1,self.walk_len+1):
            temporal_neighbor=self.get_temporal_neighbor(
                tar=cur_node,
                tar_t=cur_t,
                n_neighbor=n_candidate
            )
            neighbor=temporal_neighbor["neighbor"] # [B*M,n_candidate]
            neighbor_t=temporal_neighbor["neighbor_t"] # [B*M,n_candidate]
            neighbor_edge=temporal_neighbor["neighbor_edge"] # [B*M,n_candidate]
            neighbor_mask=temporal_neighbor["neighbor_mask"] # [B*M,n_candidate]

            valid=neighbor_mask&active.unsqueeze(1) # [B*M,n_candidate]
            has_history=valid.any(dim=1) # [B*M,]

            logits=alpha*(
                neighbor_t.to(torch.float64)
                -cur_t.unsqueeze(1).to(torch.float64)
            ) # [B*M,n_candidate]
            logits=logits.masked_fill(~valid,-torch.inf) # valid=False인 후보의 Logit을 -inf로 설정, Softmax에서 exp(-inf)=0

            # 후보가 없는 walk 처리
            logits=torch.where(
                has_history.unsqueeze(1),
                logits,
                torch.zeros_like(logits)
            )

            # 후보 link 중 하나 샘플링
            # Categorical: 각 행의 Logit들을 확률분포로 변환한 뒤, 해당 분포에서 인덱스 하나를 샘플링
            sampled_idx=torch.distributions.Categorical(logits=logits).sample().unsqueeze(1) # [B*M,1]

            # 선택된 link의 정보 추출
            # gather(dim=1, index=sampled_idx): 각 행에서 지정된 열의 값을 추출
            next_node=neighbor.gather(1,sampled_idx).squeeze(1) # [B*M]
            next_t=neighbor_t.gather(1,sampled_idx).squeeze(1) # [B*M]
            next_edge=neighbor_edge.gather(1,sampled_idx).squeeze(1) # [B*M]

            # 종료된 walk의 padding 처리
            # torch.where(condition, x, y): 조건이 참이면 x, 거짓이면 y를 선택
            next_node=torch.where(has_history,next_node,0) # [B*M]
            next_t=torch.where(has_history,next_t,0) # [B*M]
            next_edge=torch.where(has_history,next_edge,0) # [B*M]

            # walk 결과 저장
            walks_node[:,hop]=next_node # [B*M,m+1]
            walks_t[:,hop]=next_t # [B*M,m+1]
            walks_edge[:,hop]=next_edge # [B*M,m+1]

            # 다음 hop의 기준 갱신
            cur_node=next_node # [B*M], 새로운 탐색 시작 노드
            cur_t=next_t # [B*M], 새로운 시간 상한
            active=has_history # [B*M], 계속 샘플링할 Walk 구분

            # 모든 walk가 종료된 경우
            if not active.any().item():
                break

        return {
            "walks_node":walks_node.reshape(n_root,self.n_walk,self.walk_len+1),
            "walks_t":walks_t.reshape(n_root,self.n_walk,self.walk_len+1),
            "walks_edge":walks_edge.reshape(n_root,self.n_walk,self.walk_len+1)
        }