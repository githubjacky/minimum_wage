library(tibble)
library(magrittr)
library(haven)
library(rsample)
suppressPackageStartupMessages(library(poorman))
library(tune)
library(yardstick)


train_test_split <- function(prop = 0.8, seed = 20230225) {
    data_path <- "data/processed/training.dta"
    data <- haven::read_dta(data_path)

    # split
    set.seed(seed)
    split <- initial_split(data, prop = prop, strata = group)

    return(split)
}



tune <- function(data_train, wf, grid) {
    set.seed(20230225)
    folds <- rsample::vfold_cv(data_train, v = 5, strata = group)

    print("tuning with grid")
    grid_res <- wf |>
        tune::tune_grid(
            resamples = folds,
            grid = grid,
            metrics = yardstick::metric_set(pr_auc),
            # control=control_grid(save_workflow=T)
        )

    return(grid_res)
}


# fit on the training split and calculate the metric on testing split
test <- function(model, name, split) {
    res <- last_fit(model, split = split) |>
        collect_predictions() |>
        mutate(model = name)

    return(res)
}