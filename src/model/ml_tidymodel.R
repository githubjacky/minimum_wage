rm(list = ls(all = TRUE))
setwd("/Users/jacky/Documents/research/RA/Yang/minimum_wage_ML/data")

################################################################################
# collect the data, train test split and create the formula
################################################################################

library(tidyverse)
library(tidymodels)



# helper function for tuning
################################################################################


################################################################################
# elastic net
################################################################################
################################################################################


################################################################################
# random forest
################################################################################
rf_spec <- rand_forest(
    mtry = tune(), # number of predictors
    trees = tune(), # number of trees
    min_n = tune() # stop split when the sample < min_n
) %>%
    set_mode("classification") %>%
    set_engine("ranger")

wf <- create_workflow(rf_spec)
grid <- expand.grid(
    mtry = c(4, 5),
    trees = c(100, 200, 300, 400, 500),
    min_n = c(200, 300, 400, 500, 600, 700)
)
rf_tune <- tune_result(wf, grid)
autoplot(rf_tune)
random_forest <- wf %>%
    finalize_workflow(select_best(rf_tune, metric = "pr_auc"))

# random_forest <- update(rf_spec, mtry=4, trees=200, min_n=400)

random_forest_test <- test(random_forest, "Random Forest")
################################################################################


################################################################################
# xgboost
################################################################################
xgb_spec <- boost_tree(
    mtry = tune(), # number of predictors
    trees = tune(), # number of trees
    min_n = tune(), # stop split when the sample < min_n(minimal node size)
    tree_depth = tune(), # usually: 3 ~ 10
    learn_rate = tune(),
    loss_reduction = tune(), # stop when loss reduction < loss_reduction(minimal loss reduction)
    sample_size = tune(),
    stop_iter = tune() # stop when iterations > stop_iter(iterations before stopping)
) %>%
    set_mode("classification") %>%
    set_engine("xgboost")


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
# 4. tune the learn_rate
wf <- create_workflow(xgb_spec)
grid <- expand.grid(
    mtry = 5,
    trees = 400,
    min_n = 300,
    tree_depth = 9,
    # learn_rate=0.3,
    loss_reduction = 1e-2,
    sample_size = 0.9,
    stop_iter = 800,

    # mtry=c(3, 4, 5),
    # trees=c(50, 200, 400, 600),
    # min_n=c(200, 300, 400, 500, 600, 700),
    # tree_depth=c(3, 5, 7, 9, 11)
    learn_rate = seq(1e-1, 1, length.out = 10)
    # loss_reduction=c(1e-1, 1e-2, 1e-3, 1e-4, 1e-5),
    # sample_size=c(0.3, 0.5, 0.6, 0.7, 0.8, 0.9)
    # stop_iter=c(50, 200, 400, 800)
)
xgb_tune <- tune_result(wf, grid)
autoplot(xgb_tune)

xgboost <- wf %>%
    finalize_workflow(select_best(xgb_tune, metric = "pr_auc"))

# xgboost <- update(
#     xgb_spec, mtry=5, trees=400, min_n=300, tree_depth=9, learn_rate=0.3,
# .    loss_reduction=1e-2, sample_size=0.9, stop_iter=800
# )

xgboost_test <- test(xgboost, "XGBoost")
################################################################################


################################################################################
# model selection
bind_rows(elastic_net_test, random_forest_test, xgboost_test) %>%
    group_by(model) %>%
    pr_curve(group, .pred_test:.pred_control2) %>%
    autoplot()
################################################################################
