docker run --rm -it \
	--name train__$1 \
	-v $PWD/notebooks/:/minimum_wage/notebooks \
	-v $PWD/src/:/minimum_wage/src \
	-v $PWD/data/tune:/minimum_wage/data/tune \
	-v $PWD/data/model:/minimum_wage/data/model \
	0jacky/minimum_wage:env \
	Rscript src/model/$1.R
