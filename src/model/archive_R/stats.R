rm(list=ls(all=TRUE))
setwd("/Users/jacky/Documents/research/RA/Yang/minimum_wage_ML/data")


library(tidyverse)
library(haven)
library(gtools)                                    # Install gridExtra package
library("gridExtra")

data_path <- "wdata/training.dta"
data <- as_tibble(haven::read_dta(data_path)) %>%
    select("countyname", "sex", "martial", "educat", "agecat", "group")

# set the factors for dependent variable
for(col in names(data)){
    data[[col]] <- as.factor(data[[col]])
}
levels(data$group) <- c("test", "control1", "control2")

# distribution of the predictors
p1 <- data %>%
    select(-c(group, countyname)) %>%
    gather() %>%
    ggplot(aes(value)) +
    facet_wrap(~key, scales="free") +
    geom_histogram(stat="count", fill="steelblue")

p2 <- data %>%
    select(countyname) %>%
    ggplot(aes(countyname)) +
    geom_histogram(stat="count", fill="steelblue") +
    theme(axis.text.x=element_text(family="Noto Serif CJK TC"))

cowplot::plot_grid(p1, p2, nrow=2)

# distribution of the outcome variable
data %>%
    select(group) %>%
    ggplot(aes(group)) +
    geom_histogram(stat="count")

name <- c("sex", "martial", "educat")
for(i in name){
    data[[i]] <- as.numeric(data[[i]])
}
var <- gtools::combinations(
    n=length(name),
    r=2,
    name, 
    repeats.allowed=FALSE
)
plots <- list()
for(i in 1:nrow(var)){
    col1 <- var[i, 1]
    col2 <- var[i, 2]
    p <- data %>%
        ggplot(aes(
            get(col1), get(col2), color=data$group, shape=data$group
        )) +
        geom_point() +
        xlab(col1) + ylab(col2)
    plots[[i]] <- p
}
multiplot(plots)
