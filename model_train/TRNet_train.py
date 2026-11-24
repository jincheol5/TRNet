import numpy as np
import torch
import torch.nn as nn
import mlflow
from tqdm import tqdm
from torch.utils.data import DataLoader
from model import TRNet
from utils import TrainUtils,EarlyStopper,MetricUtils

class TRNet_Trainer:
    @staticmethod
    def train(
            model:TRNet,
            train_loader:DataLoader,
            val_loader:DataLoader,
            val_sample_loader:list,
            train_TR_label:torch.Tensor,
            **kwargs
        ):
        """
        Set GPU, Optimizer
        """
        if torch.cuda.is_available():
            device=torch.device("cuda")
        elif torch.backends.mps.is_available():
            device=torch.device("mps")
        else:
            device=torch.device("cpu")
        model=model.to(device)

        if kwargs["optimizer"]=="adam":
            optimizer=torch.optim.Adam(
                model.parameters(),
                lr=kwargs["lr"]
            )
        else:
            optimizer=torch.optim.SGD(
                model.parameters(),
                lr=kwargs["lr"]
            )

        """
        Set Early Stopper
        """
        if kwargs["early_stop"]:
            early_stop=EarlyStopper(patience=kwargs["patience"])

        """
        Train model
        """
        best_metrics=None
        for epoch in tqdm(range(kwargs["epoch"]),desc=f"Model Training..."):
            ### init info 
            model.info.init_info()

            ### Epoch마다 train_sample_loader 생성
            train_sample_loader=TrainUtils.get_TR_sample_loader(
                graph_loader=train_loader,
                n_sample=kwargs["n_sample"],
                n_pair=kwargs["n_pair"],
                candidates=kwargs["train_candidates"],
                TR_label=train_TR_label
            )

            # epoch별 train loss 기록
            epoch_train_loss_list=[]

            model.train()
            for batch_event,batch_sample in tqdm(
                    zip(train_loader,train_sample_loader),
                    total=len(train_loader),
                    desc=f"Training epoch {epoch+1}..."
                ):
                ### update info of batch_eventstream
                event_src,event_dst,event_t,event_edge=batch_event
                event_src=event_src.to(device)
                event_dst=event_dst.to(device)
                event_t=event_t.to(device)
                event_edge=event_edge.to(device)
                model.update_info(
                    src=event_src,
                    dst=event_dst,
                    event_t=event_t
                )

                ### TR Sample
                src=batch_sample["src"]
                dst=batch_sample["dst"]
                label=batch_sample["label"]
                src=src.to(device)
                dst=dst.to(device)
                label=label.to(device)

                ### Forward
                pred_logit=model(
                    src=src,
                    dst=dst
                ) # [B,]

                ### Loss
                criterion=nn.BCEWithLogitsLoss()
                loss=criterion(pred_logit,label)
                epoch_train_loss_list.append(loss.item())

                ### Backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                ### detach info
                model.info.detach_info()

            """
            Validate Model
            """
            val_result=TRNet_Trainer.validate(
                model=model,
                val_loader=val_loader,
                val_sample_loader=val_sample_loader,
                **kwargs
            )
            val_acc=val_result["acc"]
            val_ap=val_result["ap"]
            val_auc=val_result["auc"]
            print(f"{epoch+1} epoch  Validate ACC: {val_acc}")
            print(f"{epoch+1} epoch  Validate AP: {val_ap}")
            print(f"{epoch+1} epoch  Validate AUC: {val_auc}")

            """
            MLFlow tracking
            """
            if kwargs["use_mlflow"]:
                # epoch train loss 평균 계산 
                train_loss=np.mean(epoch_train_loss_list)
                mlflow.log_metrics({
                    "train_loss":train_loss,
                    "val_loss":val_result["loss"],
                    "val_acc":val_result["acc"],
                    "val_ap":val_result["ap"],
                    "val_auc":val_result["auc"]
                },step=epoch+1)

            """
            Check Early Stop
            """
            val_loss=val_result["loss"]

            # check best_metrics
            if np.isfinite(val_loss) and (
                    best_metrics is None or val_loss<best_metrics["best_val_loss"]
                ):
                best_metrics={
                    "best_epoch":epoch+1,
                    "best_val_loss":val_loss,
                    "best_val_acc":val_acc,
                    "best_val_ap":val_ap,
                    "best_val_auc":val_auc
                }
            print(f"{epoch+1} epoch Validate Loss: {val_loss}")

            # check early_stop
            if kwargs["early_stop"]:
                pre_model=early_stop(
                    val_loss=val_loss,
                    model=model
                )
                if early_stop.early_stop:
                    model=pre_model
                    print(f"Early Stop in epoch {epoch+1}")
                    break

        # early_stop이 max_epoch 내에 안 끝난 경우 최적 모델 가중치 설정
        if kwargs["early_stop"] and early_stop.best_state is not None:
            model.load_state_dict(early_stop.best_state)

        # mlflow로 기록하는 경우 best_epoch (min_val_loss)에서의 metric 기록 저장
        if kwargs["use_mlflow"]:
            if best_metrics is not None:
                mlflow.log_metrics(best_metrics,step=best_metrics["best_epoch"])

        return model

    @staticmethod
    def validate(
            model:TRNet,
            val_loader:DataLoader,
            val_sample_loader:list,
            **kwargs
        ):
        """
        Set GPU
        """
        if torch.cuda.is_available():
            device=torch.device("cuda")
        elif torch.backends.mps.is_available():
            device=torch.device("mps")
        else:
            device=torch.device("cpu")
        model=model.to(device)
        model.graph.to_device(device=device)
        model.eval()

        """
        Compute validate loss and acc
        """
        loss_list=[]
        pred_logit_list=[]
        label_list=[]
        with torch.no_grad():
            for batch_event,batch_sample in tqdm(
                    zip(val_loader,val_sample_loader),
                    total=len(val_loader),
                    desc=f"Validate..."
                ):
                ### update info of batch_eventstream
                event_src,event_dst,event_t,event_edge=batch_event
                event_src=event_src.to(device)
                event_dst=event_dst.to(device)
                event_t=event_t.to(device)
                event_edge=event_edge.to(device)
                model.update_info(
                    src=event_src,
                    dst=event_dst,
                    event_t=event_t
                )

                ### TR Sample
                src=batch_sample["src"]
                dst=batch_sample["dst"]
                label=batch_sample["label"]
                src=src.to(device)
                dst=dst.to(device)
                label=label.to(device)

                ### Forward
                pred_logit=model(
                    src=src,
                    dst=dst
                ) # [B,]

                ### Loss
                criterion=nn.BCEWithLogitsLoss()
                batch_loss=criterion(pred_logit,label)
                loss_list.append(batch_loss)

                # Collect all samples on CPU for dataset-level metrics.
                pred_logit_list.append(pred_logit.detach().cpu())
                label_list.append(label.detach().cpu())

        pred_logit=torch.cat(pred_logit_list)
        label=torch.cat(label_list)
        return {
            "loss":torch.stack(loss_list).mean().item(),
            "acc":MetricUtils.compute_ACC(pred_logit=pred_logit,label=label),
            "ap":MetricUtils.compute_AP(pred_logit=pred_logit,label=label),
            "auc":MetricUtils.compute_AUC(pred_logit=pred_logit,label=label),
        }

    @staticmethod
    def test(
            model:TRNet,
            test_loader:DataLoader,
            test_sample_loader:list,
            **kwargs
        ):
        """
        Set GPU
        """
        if torch.cuda.is_available():
            device=torch.device("cuda")
        elif torch.backends.mps.is_available():
            device=torch.device("mps")
        else:
            device=torch.device("cpu")
        model=model.to(device)
        model.graph.to_device(device=device)
        model.eval()

        """
        Compute test acc
        """
        pred_logit_list=[]
        label_list=[]
        with torch.no_grad():
            for batch_event,batch_sample in tqdm(
                    zip(test_loader,test_sample_loader),
                    total=len(test_loader),
                    desc=f"Test..."
                ):
                ### update info of batch_eventstream
                event_src,event_dst,event_t,event_edge=batch_event
                event_src=event_src.to(device)
                event_dst=event_dst.to(device)
                event_t=event_t.to(device)
                event_edge=event_edge.to(device)
                model.update_info(
                    src=event_src,
                    dst=event_dst,
                    event_t=event_t
                )

                ### TR Sample
                src=batch_sample["src"]
                dst=batch_sample["dst"]
                label=batch_sample["label"]
                src=src.to(device)
                dst=dst.to(device)
                label=label.to(device)

                ### Forward
                pred_logit=model(
                    src=src,
                    dst=dst
                ) # [B,]

                # Collect all samples on CPU for dataset-level metrics.
                pred_logit_list.append(pred_logit.detach().cpu())
                label_list.append(label.detach().cpu())

        pred_logit=torch.cat(pred_logit_list)
        label=torch.cat(label_list)
        return {
            "acc":MetricUtils.compute_ACC(pred_logit=pred_logit,label=label),
            "ap":MetricUtils.compute_AP(pred_logit=pred_logit,label=label),
            "auc":MetricUtils.compute_AUC(pred_logit=pred_logit,label=label),
        }