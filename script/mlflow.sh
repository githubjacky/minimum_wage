docker run --rm \
	--name mlfow_ui \
	-v $PWD/mlruns:/minimum_wage/mlruns \
	-p 5050:5050 \
	0jacky/minimum_wage:env \
	bash -c 'mlflow ui -p 5050 -h 0.0.0.0'
