library(parsnip)
suppressPackageStartupMessages(library(recipes))
library(ggplot2)
library(workflows)
source("src/model/utils.R")

main <- function() {
    split <- train_test_split()

    spec <- boost_tree(
        mtry = tune(), # number of predictors
        trees = tune(), # number of trees
        min_n = tune(), # stop split when the sample < min_n(minimal node size)
        tree_depth = tune(), # usually: 3 ~ 10
        learn_rate = tune(),
        loss_reduction = tune(), # stop when loss reduction < loss_reduction(minimal loss reduction)
        sample_size = tune(),
        stop_iter = tune() # stop when iterations > stop_iter(iterations before stopping)
    ) |>
        set_mode("classification") |>
        set_engine("xgboost")

    recipe <- recipe(
        group ~ countycat + sex + martial + educat + agecat,
        data = split
    ) |>
        step_num2factor(
            group,
            levels = c("test", "control1", "control2")
        )

    wf <- workflow(recipe, spec)
    res <- tune(
        rsample::training(split),
        wf,
        grid = 3
    )
    tune_plot <- tune::autoplot(res, type = "marginals")
    ggsave(tune_plot, filename = "plot/tune/xbooost/tune.png")

    model <- tune::finalize_workflow(wf, tune::select_best(res, metric = "pr_auc"))
}

main()


# tuning strategy:
# 1.  tune min_n, tree depth
#    other default parameters:
#        - mtry = 80%of the predictors
#        - trees: 200
#        - loss_reduction = 0
#        - sample_size: 80% of the sample size
#        - stop_iter = 2000(should be large enought)
# 2. tune mtry, sample_size
# 3. tune trees, loss_reduction, stop_iter
