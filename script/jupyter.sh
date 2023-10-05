docker run --rm \
	--name jupyterlab \
	-p 8888:8888 \
	-p 5050:5050 \
	-v $PWD/notebooks:/minimum_wage/notebooks \
	-v $PWD/src:/minimum_wage/src \
	-v $PWD/data/processed:/minimum_wage/data/processed \
	-v $PWD/plot:/minimum_wage/plot \
	-v $PWD/model:/minimum_wage/model \
	-v $PWD/mlruns:/minimum_wage/mlruns \
	0jacky/minimum_wage:env \
	jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root
