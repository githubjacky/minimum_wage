docker run --rm \
	--name train__$1 \
	-v $PWD/notebooks/:/minimum_wage/notebooks \
	-v $PWD/src/:/minimum_wage/src \
	-v $PWD/data/processed/:/minimum_wage/data/processed \
	-v $PWD/plot:/minimum_wage/plot \
	-v $PWD/model:/minimum_wage/model \
	-v $PWD/mlruns:/minimum_wage/mlruns \
	0jacky/minimum_wage:env \
	Rscript src/model/$1.R
