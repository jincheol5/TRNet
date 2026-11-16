#!/bin/bash
set -e

python -m app.dataset.save_sample_loader --dataset_name CollegeMsg --purpose val --seed 1 --batch_size 200

python -m app.dataset.save_sample_loader --dataset_name CollegeMsg --purpose test --seed 1 --batch_size 200