library(doParallel)
library(haven)
library(magrittr)
library(mlflow)
suppressPackageStartupMessages(library(poorman))
library(rsample)
library(tibble)
library(tune)
library(yardstick)


train_test_split <- function(prop = 0.8, seed = 20230225) {
    data_path <- "data/processed/training.dta"
    data <- haven::read_dta(data_path)
    data$countyname <- toString(data$countyname)

    # split
    set.seed(seed)
    split <- initial_split(data, prop = prop, strata = group)

    return(split)
}


tune <- function(data_train, wf, grid) {
    set.seed(20230225)
    folds <- vfold_cv(data_train, v = 5, strata = group)


    cl <- makeCluster(5)
    registerDoParallel(cl)

    grid_res <- wf |>
        tune_grid(
            resamples = folds,
            grid = grid,
            metrics = metric_set(pr_auc, recall, precision, f_meas, roc_auc),
            control = control_grid(parallel_over = "resamples")
        )

    return(grid_res)
}

log_val <- function(wf, res_tbl, model_name, exper_name, tune_strat) {
    mlflow_server(
        file_store = "mlruns",
        host = "0.0.0.0",
        port = 5050
    )
    experiment_id <- mlflow_set_experiment(experiment_name = exper_name)
    print(paste("experiment id:", experiment_id))

    spec <- workflows::extract_spec_parsnip(wf)
    param_name <- names(spec$args)

    for (i in seq(1, nrow(res_tbl), by = 5)) {
        run <- mlflow_start_run(experiment_id = experiment_id, nested = TRUE)
        run_id <- run$run_id

        mlflow_set_tag("model", model_name, run_id = run_id)
        mlflow_set_tag("tune strategy", tune_strat, run_id = run_id)
        mlflow_set_tag("stage", "val", run_id = run_id)

        for (name in param_name) {
            mlflow_log_param(name, res_tbl[i, name], run_id = run_id)
        }
        mlflow_log_metric("f_meas", res_tbl$mean[i], run_id = run_id)
        mlflow_log_metric("pr_auc", res_tbl$mean[i + 1], run_id = run_id)
        mlflow_log_metric("precision", res_tbl$mean[i + 2], run_id = run_id)
        mlflow_log_metric("recall", res_tbl$mean[i + 3], run_id = run_id)
        mlflow_log_metric("roc_auc", res_tbl$mean[i + 4], run_id = run_id)

        mlflow_end_run(status = "FINISHED", run_id = run_id)
    }

    return(experiment_id)
}

log_test <- function(wf, split, grid_res, model_name, experiment_id) {
    best_param <- select_best(grid_res, metric = "pr_auc")
    model <- finalize_workflow(wf, best_param)

    spec <- workflows::extract_spec_parsnip(wf)
    param_name <- names(spec$args)

    res_tbl <- last_fit(
        model,
        split = split,
        metrics = metric_set(pr_auc, recall, precision, f_meas, roc_auc)
    )$.metrics[[1]] |> arrange(.metric)

    mlflow_server(
        file_store = "mlruns",
        host = "0.0.0.0",
        port = 5050
    )
    run <- mlflow_start_run(experiment_id = experiment_id, nested = TRUE)
    run_id <- run$run_id

    for (name in param_name) {
        mlflow_log_param(name, best_param[1, name], run_id = run_id)
    }

    mlflow_set_tag("model", model_name, run_id = run_id)
    mlflow_set_tag("stage", "test", run_id = run_id)

    mlflow_log_metric("f_meas", res_tbl$.estimate[1], run_id = run_id)
    mlflow_log_metric("pr_auc", res_tbl$.estimate[2], run_id = run_id)
    mlflow_log_metric("precision", res_tbl$.estimate[3], run_id = run_id)
    mlflow_log_metric("recall", res_tbl$.estimate[4], run_id = run_id)
    mlflow_log_metric("roc_auc", res_tbl$.estimate[5], run_id = run_id)

    mlflow_end_run(status = "FINISHED", run_id = run_id)

    return(model)
}



# fit on the training split and calculate the metric on testing split
test <- function(model, name, split) {
    res <- last_fit(model, split = split) |>
        collect_predictions() |>
        mutate(model = name)

    return(res)
}
