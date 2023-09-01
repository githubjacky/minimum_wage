library(parsnip)
suppressPackageStartupMessages(library(recipes))

library(workflows)
source("src/model/utils.R")


main <- function() {
    split <- train_test_split()

    spec <- multinom_reg(
        penalty = tune(), # regularization
        mixture = tune() # alpha: ratio of L1 and L2 regularization
    ) |>
        parsnip::set_mode("classification") |>
        parsnip::set_engine("glmnet")

    recipe <- recipe(
        group ~ countycat + sex + martial + educat + agecat,
        data = split
    ) |>
        step_num2factor(
            group,
            levels = c("test", "control1", "control2")
        )

    wf = workflow(recipe, spec)
    res <- tune(
        rsample::training(split),
        wf,
        grid = 3,
    )
    elastic_net <- finalize_workflow(wf, select_best(res, metric = "pr_auc"))
}


main()
