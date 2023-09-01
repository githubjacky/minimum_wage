docker run --rm \
	--name jupyterlab \
	-p 8888:8888 \
	-v $PWD/notebooks/:/minimum_wage/notebooks \
	-v $PWD/src/:/minimum_wage/src \
	-v $PWD/data/tune:/minimum_wage/data/tune \
	-v $PWD/data/model:/minimum_wage/data/model \
	0jacky/minimum_wage:env \
	jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root
