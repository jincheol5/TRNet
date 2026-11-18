#!/bin/bash
set -e

python -m app.experiment.train_model --model_name TGAT --dataset_name CollegeMsg

python -m app.experiment.train_model --model_name TGN --dataset_name CollegeMsg

python -m app.experiment.train_model --model_name CAW-N --dataset_name CollegeMsg