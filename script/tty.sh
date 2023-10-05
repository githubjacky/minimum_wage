docker run --rm -it --name tty \
	-p 5050:5050 \
	-v $PWD/mlruns:/minimum_wage/mlruns \
	0jacky/minimum_wage:env
