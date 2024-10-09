#!/bin/bash

CONTAINER_NAME=jupyterlab
DOCKER_IMAGE=jupyter/tensorflow-notebook:latest
ENV_NAME=py311_tensorflow
WORK_DIR="/home/jovyan/work"

\docker pull $DOCKER_IMAGE

if [ "$(docker ps -q -f name=$CONTAINER_NAME)" ]; then
    echo "Container '$CONTAINER_NAME' is already running. Stopping and removing it."
    docker stop $CONTAINER_NAME
    docker rm $CONTAINER_NAME
fi

echo "Running a new '$CONTAINER_NAME' container."
docker run -d \
    --name $CONTAINER_NAME \
    -p 8888:8888 \
    -v "$PWD":$WORK_DIR \
    -e PYTHONPATH=$WORK_DIR:$PYTHONPATH \
    jupyter/tensorflow-notebook start.sh jupyter lab --NotebookApp.token='' --NotebookApp.password=''

echo "JupyterLab is running. You can access it via http://localhost:8888 in your browser."

# 컨테이너 내에서 conda 환경 및 ipykernel 설정
echo "Setting up conda environment and installing packages..."

docker exec -it $CONTAINER_NAME /bin/bash -c "
    conda create -n $ENV_NAME python=3.11 -y
    conda run -n $ENV_NAME pip install -r $WORK_DIR/requirements.txt
    conda run -n $ENV_NAME pip install ipykernel
    conda run -n $ENV_NAME python -m ipykernel install --user --name=$ENV_NAME --display-name '$ENV_NAME'
"

echo "JupyterLab is running. You can access it via http://localhost:8888 in your browser."