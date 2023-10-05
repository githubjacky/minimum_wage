FROM r-base:4.3.1 AS base
WORKDIR /minimum_wage

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install --no-install-recommends -y build-essential \
    python3 python3-pip python3-setuptools python3-dev \
    libssl-dev libcurl4-openssl-dev

RUN pip3 install mlflow jupyterlab --break-system-packages

ENV MLFLOW_PYTHON_BIN=/usr/bin/pyton3
ENV MLFLOW_BIN=/usr/local/bin/mlflow

RUN Rscript -e "install.packages('foreach')"
RUN Rscript -e "install.packages('tidymodels')"
RUN Rscript -e "install.packages('themis')"
RUN Rscript -e "install.packages('haven')"
RUN Rscript -e "install.packages('poorman')"
RUN Rscript -e "install.packages('doParallel')"
RUN Rscript -e "install.packages('iterators')"
RUN Rscript -e "install.packages('parallel')"
RUN Rscript -e "install.packages('glmnet')"
RUN Rscript -e "install.packages('ranger')"
RUN Rscript -e "install.packages('mlflow')"
RUN Rscript -e "install.packages('IRkernel'); IRkernel::installspec(user=FALSE)"
RUN Rscript -e "install.packages('finetune')"
RUN Rscript -e "install.packages('DataExplorer')"
# RUN Rscript -e "install.packages('xgboost')"


EXPOSE 8888
EXPOSE 5000


CMD ["bash"]
