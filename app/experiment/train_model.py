import argparse
import mlflow
from torch.utils.data import DataLoader
from model import *
from model_train import *
from utils import *
from .config_info import *

def main(**kwargs):
    """
    0. Init Setting
    """
    ### unpack kwargs
    model_name=kwargs["model_name"]
    seed=kwargs["seed"]
    lr=kwargs["lr"]
    dataset_name=kwargs["dataset_name"]
    use_mlflow=kwargs["use_mlflow"]
    experiment_name=kwargs["experiment_name"]
    save_model=kwargs["save_model"]

    ### set seed
    TrainUtils.set_seed(seed=seed)

    """
    1. Data
    """
    ### load data
    data=DataUtils.preprocess_SNAP_dataset(dataset_name=dataset_name,use_edge_ft=False)
    graph_df=data["graph_df"]
    node_dim=172
    edge_dim=172

    ### load train_TR_result
    batch_size=TRAIN_CONFIG["batch_size"]
    train_TR_result=DataUtils.load_TR_result(
        dataset_name=dataset_name,
        purpose="train",
        batch_size=batch_size
    )

    ### set train_loader and candidates, val_loader // load val_sample_loader
    train_df,val_df,_=TrainUtils.split_graph_df(graph_df=graph_df)
    train_dataset=TemporalGraphDataset(graph_df=train_df)
    train_loader=DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=False)
    train_candidates=train_dataset.src.tolist()+train_dataset.dst.tolist()

    val_dataset=TemporalGraphDataset(graph_df=val_df)
    val_loader=DataLoader(dataset=val_dataset,batch_size=batch_size,shuffle=False)
    val_sample_loader=DataUtils.load_sample_loader(
        dataset_name=dataset_name,
        purpose="val",
        seed=seed,
        batch_size=batch_size
    )

    """
    2. Model
    """
    ### set model
    common_model_config={
        "model_name":model_name,
        "node_dim":node_dim,
        "edge_dim":edge_dim,
        **COMMON_MODEL_CONFIG,
    }
    specific_model_config=SPECIFIC_MODEL_CONFIG[model_name]
    model_config=common_model_config|specific_model_config

    # 모델 생성자에 필요한 파라미터만 추출
    init_param_config={
        key:model_config[key]
        for key in MODEL_PARAM[model_name]
    }
    model=MODEL_REGISTRY[model_name](**init_param_config)

    # set model graph
    graph=TrainUtils.set_model_graph(
        model_name=model_name,
        graph_df=graph_df,
        **specific_model_config
    )
    model.set_graph(graph=graph)

    """
    3. Train
    """
    train_config={
        **TRAIN_CONFIG,
        "seed":seed,
        "lr":lr,
        **SAMPLING_CONFIG,
        "train_candidates":train_candidates,
        "use_mlflow":use_mlflow
    }
    model_trainer=TRAINER_REGISTRY[model_name]

    if use_mlflow:
        ### set mlflow 
        MLFlowUtils.setup(experiment_name=experiment_name)
        run_name=f"{model_name}_{dataset_name}_B{batch_size}_S{seed}_LR{lr}"
        with mlflow.start_run(run_name=run_name) as run:
            # log hyperparameters
            dataset_config={"dataset_name":dataset_name}
            mlflow.log_params(dataset_config|model_config|train_config)

            ### train model
            model=model_trainer.train(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                val_sample_loader=val_sample_loader,
                train_TR_label=train_TR_result["label"],
                **train_config
            )

            ### save model
            if save_model:
                MLFlowUtils.save_model(model=model,artifact_name=model_name)

            # mlflow run_id
            run_id=run.info.run_id
        print(f"Training completed: {run_name}")
        print(f"Run ID: {run_id}")
    else:
        ### train model
        model=model_trainer.train(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            val_sample_loader=val_sample_loader,
            train_TR_label=train_TR_result["label"],
            **train_config
        )
        print(f"Training completed!")

if __name__=="__main__":
    """
    Execute app
    """
    parser=argparse.ArgumentParser()
    parser.add_argument("--model_name",
        type=str,
        choices=list(MODEL_REGISTRY),
        default=f"TGAT"
    )
    parser.add_argument("--seed",type=int,choices=[1,2,3],default=1)
    parser.add_argument("--lr",type=float,choices=[0.0001,0.0005],default=0.0001)
    parser.add_argument("--dataset_name",
        type=str,
        choices=DATASET_REGISTRY,
        default="CollegeMsg"
    )
    parser.add_argument("--use_mlflow",type=int,default=0)
    parser.add_argument("--experiment_name",type=str,default=f"TRNet")
    parser.add_argument("--save_model",type=int,default=0)
    args=parser.parse_args()
    main(**vars(args))