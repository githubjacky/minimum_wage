FROM r-base:4.3.1
WORKDIR /minimum_wage

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install --no-install-recommends -y build-essential \
    python3 python3-pip python3-setuptools python3-dev

RUN pip3 install jupyterlab --break-system-packages


RUN Rscript -e "install.packages('tidymodels')"
RUN Rscript -e "install.packages('haven')"
RUN Rscript -e "install.packages('poorman')"
RUN Rscript -e "install.packages('doParallel')"
RUN Rscript -e "install.packages('foreach')"
RUN Rscript -e "install.packages('iterators')"
RUN Rscript -e "install.packages('parallel')"
RUN Rscript -e "install.packages('glmnet')"
RUN Rscript -e "install.packages('xgboost')"
RUN Rscript -e "install.packages('ranger')"
RUN Rscript -e "install.packages('mlflow')"
RUN Rscript -e "install.packages('IRkernel'); IRkernel::installspec(user=FALSE)"

# COPY data/raw data/raw
# COPY data/processed data/processed


EXPOSE 8888

CMD ["bash"]
