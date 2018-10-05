#!/bin/bash
# Workspace
WORK_SPACE=$(pwd)
MODEL_DIR="${WORK_SPACE}/Model"

cd im2txt
# Run a TensorBoard server.
tensorboard --logdir="${MODEL_DIR}"