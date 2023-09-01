rm(list=ls(all=TRUE))
setwd("/Users/jacky/Documents/research/RA/Yang/minimum_wage_ML/data")

library(haven)
library(glmnet)
library(ggplot2)
library(cowplot)
library(caret)
library(PRROC)
library(tidyverse)
library(xgboost)

################################################################################
# data preprocessing
################################################################################
data_path <- "wdata/training.dta"
data <- as_tibble(haven::read_dta(data_path))

# set the factors for dependent variable
data$group <- as.factor(data$group)  # to perform classification task in `caret::train`


# train test split
set.seed(20230225)
split_idx <- caret::createDataPartition(data$group, p=0.75, list=F)
train_split <- data[split_idx, ]
test_split <- data[-split_idx, ]
################################################################################


################################################################################
# distribution of the dependent and indepenedent variables
create_plot <- function(column) {
    p1 <- ggplot(train_split, aes(x=get(column))) +
          geom_histogram(stat="count") +
          xlab(column)
    p2 <- ggplot(test_split, aes(x=get(column))) +
          geom_histogram(stat="count") +
          xlab(column)
    
    plots <- cowplot::plot_grid(p1, p2, ncol=2)
    return(plots)
}

plots <- lapply(
    c("countyname", "sex", "agecat", "martial", "educat"), 
    function(column){create_plot(column)}
)
cowplot::plot_grid(plotlist=plots, nrow=5)

create_plot("group")
################################################################################


################################################################################
# some utility function for the elastic net and random forest's training process
praucSummary <- function(data, lev=NULL, model=NULL){
    idx_control <- data$obs == lev[1]
    idx_test <- data$obs == lev[2]
    
    # precision-recall curve
    the_curve <- PRROC::pr.curve(
        # data$test is the predicted probability in test group
        data$test[idx_test],  # predicted probability of the test group(positive class) being classified as the test 
        data$test[idx_control],  # predicted probability of the control group(negative class) being classified as the test
        curve = F
    )
    
    precision <- caret::precision(data$pred, reference=data$obs, relevant=lev[2])
    recall <- caret::recall(data$pred, reference=data$obs, relevant=lev[2])
    
    out <- c(the_curve$auc.integral, precision, recall)
    names(out) <- c("AUPRC", "Precision", "Recall")
    out
}

trControl <- caret::trainControl(
    method="cv",
    number=5,
    summaryFunction=praucSummary,
    classProbs=T
)

fitControl <- caret::trainControl(
    method="none",
    classProbs=T
)

tuneplot <- function(x, probs=.90) {
    ggplot(x) +
        coord_cartesian(ylim=c(quantile(x$results$AUPRC, probs=probs), max(x$results$AUPRC))) +
        theme_bw()
}

train_evaluate <- function(caret_fit) {
    best <- which(rownames(caret_fit$results) == rownames(caret_fit$bestTune))
    best_result <- caret_fit$results[best, ]
    for (i in colnames(best_result)){
        cat(i, ": ", best_result[, i], "\n")
    }
}

test_evaluate <- function(caret_fit, ref) {
    pred <- predict(caret_fit, ref)
    res <- caret::confusionMatrix(
        pred, 
        reference=as.factor(ref$group),
        positive="test",
        mode="prec_recall"
    )
    
    idx_test <- ref$group == "test"
    idx_control <- ref$group == "control"
    pred <- predict(caret_fit, ref, type="prob")
    the_curve <- PRROC::pr.curve(
        pred$test[idx_test],
        pred$test[idx_control],
        curve=F
    )
    cat("AUPRC: ", the_curve$auc.integral, "\n")
    cat("Precision: ", res$byClass[["Precision"]], "\n")
    cat("Recall: ", res$byClass[["Recall"]], "\n")
    cat("F1-score: ", res$byClass[["F1"]], "\n")
    cat("Accuracy: ", res$overall[["Accuracy"]], "\n")
}
################################################################################


################################################################################
# elastic net
grid <- expand.grid(
    alpha=c(0.3, 0.4, 0.5, 0.6, 0.7),
    lambda=seq(0.01, 0.03, length.out=100)
)
elastic_net <- caret::train(
    group ~ sex+age+mar+school+edu+work_exp, data=train_split,
    method="glmnet",
    trControl=trControl,
    metric="AUPRC",
    tuneGrid=grid
)
tuneplot(elastic_net)

# alpha=0.7, lambda=0.0199
# AUPRC: 0.4906, Precision: 0.6445, Recall: 0.1346
train_evaluate(elastic_net)
# AUPRC: 0.4953, Precision: 0.6649, Recall: 0.1309
test_evaluate(elastic_net, test_split)
################################################################################


################################################################################
# random forest
set.seed(20230225)
grid <- expand.grid(
    mtry=c(2, 4, 6),  # number of variables randomly sampled as candidates at each split
    splitrule="gini",
    min.node.size=c(200, 300, 500, 700, 1000)
)
random_forest <- caret::train(
    group ~ sex+age+mar+school+edu+work_exp, data=train_split,
    method="ranger",
    trControl=trControl,
    metric="AUPRC",
    tuneGrid=grid
)
tuneplot(random_forest)

# mtry=2, splitrule="gini", min.node.size=700
# AUPRC: 0.5520, Precision: 0.6290, Recall: 0.3154
train_evaluate(random_forest)
# AUPRC: 0.5506, Precision: 0.6319, Recall: 0.3132
test_evaluate(random_forest, test_split)
################################################################################


################################################################################
# xgboost
set.seed(20230225)
grid <- expand.grid(
    nrounds=seq(1800, 3000, by=50),
    eta=c(0.0001, 0.003, 0.004),
    max_depth=5,
    gamma=0.5,
    colsample_bytree= 1,
    min_child_weight=2,
    subsample=0.17
)
xgb <- caret::train(
    group ~ sex+age+mar+school+edu+work_exp, data=train_split,
    method="xgbTree",
    trControl=trControl,
    metric="AUPRC",
    tuneGrid=grid
)
tuneplot(xgb)

# nrounds: 2000, eta: 0.004, max_depth: 5, gamma: 0.5, colsample_bytree: 1
# min_child_weight: 2 subsample: 0.7
# AUPRC: 0.5525, Precision: 0.6220, Recall: 0.3271
train_evaluate(xgb)
# AUPRC: 0.5513, Precision: 0.6272, Recall: 0.3251
test_evaluate(xgb, test_split)
################################################################################



################################################################################