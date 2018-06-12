#!/usr/bin/env bash
$model_type=$1
python freeze_graph.py --input_graph=../weights/$model_type/graph.pb \
    --input_checkpoint=../weights/$model_type/model.checkpoint \
    --output_graph=../weights/$model_type/freeze_graph.pb \
    --output_node_names="ranking/feature,ranking/training,ranking/score"
