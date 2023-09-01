clear all
use "training.dta", clear


* reference
// basically, the "help" command is all you need
// https://www.stata.com/meeting/italy22/slides/Italy22_Cerulli.pdf
// https://sites.google.com/view/giovannicerulli/machine-learning-in-stata


* set python excutable
// set python_exec /opt/homebrew/bin/python3


* train-test split
// split_var: generates a flag variable distinguishing the trainng and testing observations
get_train_test, dataname("training") split(.8, .2) split_var(svar) rseed(1126)

* form the target(response) and features
global y "group"
global X "countycat sex martial educat agecat"

* syntax
// c_ml_stata_cv outcome features [if] [in], mlmodel(modeltype) data_test(filename) seed(integer) 
//     [learner_options 
//      cv_options
//      other_options]

* elastic net
cap rm CV.dta
use training_train, clear
* hyper parmeter tuning
c_ml_stata_cv $y $X, mlmodel("regmult") data_test("training_test") seed(1126)  ///
    l1_ratio(.1, .3, .5, .7, .9) alpha(.00001, .0001, .001, .01)               ///
    cross_validation("CV") n_folds(10)                                         ///
    prediction("pred")
* result
// --------------------------------------------------------------------------------
// Learner: Regularized Multinomial classification
//  
// Dataset information
//  
// Target variable = "group"                   Number of features  = 5
// N. of training units = 122507               N. of testing units = 30627
// N. of used training units = 122507          N. of used testing units = 30627
// --------------------------------------------------------------------------------
//  
// Cross-validation results
//  
// Accuracy measure = rate correct matches     Number of folds = 10
// Best grid index = 5                         Optimal penalization parameter = .3
// Optimal elastic parameter = .00001          Training accuracy = .58972231
// Testing accuracy = .58804805
// Std. err. test accuracy = .00417524
// --------------------------------------------------------------------------------
//  
// Validation results
//  
// CER = classification error rate             Training CER = .41191932
// Testing CER = .40921409
//  
// --------------------------------------------------------------------------------


* random forest
* hyper parmeter tuning
cap rm CV.dta
use training_train, clear
c_ml_stata_cv $y $X, mlmodel("randomforest") data_test("training_test") seed(1126)  ///
    tree_depth(10, 15 20) max_features(4, 5) n_estimators(100, 200, 300)            ///
    cross_validation("CV") n_folds(10)                                              ///
    prediction("pred")


* boosting
* hyper parmeter tuning
c_ml_stata_cv $y $X, mlmodel("randomforest") data_test("training_test") seed(1126)  ///
    tree_depth(8, 9, 10) learning_rate(0.01, 0.3, 1.) n_estimators(300, 400, 500)   ///
    cross_validation("CV") n_folds(10)                                              ///
    prediction("pred")
