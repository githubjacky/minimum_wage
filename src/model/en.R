library(parsnip)
suppressPackageStartupMessages(library(recipes))
library(ggplot2)
library(workflows)
source("src/model/utils.R")


main <- function() {
    split <- train_test_split()

    spec <- multinom_reg(
        penalty = tune(), # regularization
        mixture = tune() # alpha: ratio of L1 and L2 regularization
    ) |>
        set_mode("classification") |>
        set_engine("glmnet")

    recipe <- recipe(
        group ~ countycat + sex + martial + educat + agecat,
        data = split
    ) |>
        step_num2factor(
            group,
            levels = c("test", "control1", "control2")
        )

    wf <- workflow(recipe, spec)
    print("start tuning")
    res <- tune(
        rsample::training(split),
        wf,
        grid = 3
    )
    tune_plot <- tune::autoplot(res, type = "marginals")
    ggsave(tune_plot, filename = "plot/tune/en/tune.png")

    model <- tune::finalize_workflow(wf, tune::select_best(res, metric = "pr_auc"))
}


main()
