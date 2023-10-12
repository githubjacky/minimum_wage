# minimum_wage
*user the machine learning algorithm to predict who is a minimum wage worker*



## install
*showcase the instruction of building process on Google Colab T4 GPU environment*

- To activate the poetry virtual environment and install the processed data
```sh
%% shell

pip install poetry
git clone https://github.com/githubjacky/minimum_wage.git


# install dependencies(permanently on your google drive)
poetry config virtualenvs.in-project true
poetry install --no-ansi

# install the processed data
poetry run dvc pull -r origin
```


- To equip with the GPU support on colab use the following command to install
dependencies:
```sh
%% shell 

!poetry run pip install --local \
    cudf-cu11 cuml-cu11 aiohttp \
    --extra-index-url=https://pypi.nvidia.com
```


- what if you want to add you own package?
1. check out the available version of the package: `!poetry add <package>`
2. modify the pyproject.toml
3. lock: `!poetry lock`
3. install: `!poetry install --no-ansi`


## activate the virtual environment
```py
import sys
sys.path.append("/content/drive/MyDrive/<your colab directory>/.venv/lib/python3.10/site-packages")
```


## run the experimental prediction
*modify the configuration under `config/` folder, including preprocess strategy and model hyperparameters etc.*
```sh
!poetry run src/model/predict.py
```
