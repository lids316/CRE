#!/bin/bash

# ============================== 环境设置 ====================================
# 禁用 SC2155 警告，允许在同一行设置和导出变量
# shellcheck disable=SC2155

# 设置 PYTHONPATH 环境变量为当前工作目录（当前文件所在的目录）
export PYTHONPATH=$(pwd)

# 设置 CUDA 可见设备
# $1 表示运行脚本时传入的第一个参数，将其作为 CUDA_VISIBLE_DEVICES 的值
CUDA_VISIBLE_DEVICES=$1 python3 main.py \
  --memory_size 10 \
  --total_rounds 5 \
  --task_name TACRED \
  --data_file data/data_with_marker_tacred.json \
  --relation_file data/id2rel_tacred.json \
  --num_of_train_samples 420 \
  --num_of_val_samples 140 \
  --num_of_test_samples 140 \
  --batch_size 32 \
  --num_of_relations 40 \
  --cache_file data/TACRED_data.pt \
  --relations_per_task 4 \
  --additional_classifier 1
