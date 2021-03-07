FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-devel

RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN conda install -y -c conda-forge opencv \
    && conda clean -ya

RUN pip install \
    GitPython \
    kornia==0.4.0 \
    loguru \
    mlconfig \
    mlflow \
    scipy \
    tqdm \
    && rm -rf ~/.cache/pip

ENV LC_ALL C.UTF-8
ENV LANG C.UTF-8
ENV LANGUAGE C.UTF-8

CMD ["mlflow", "server", "--host", "0.0.0.0"]

WORKDIR /workspace
