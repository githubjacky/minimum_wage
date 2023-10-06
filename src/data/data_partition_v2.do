clear
local data "/Users/jackyyeh/github/minimum_wageML/data/raw/mu1991to2020.dta"
use `data'

********************************************************************************

* create training data

********************************************************************************
* county_district: 市縣 ＋ 鄉鎮市區
* countyname: 市縣
* mar: 婚姻狀態（未婚、有配偶（含與人同居）、離婚分居、配偶死亡）
* edu: 教育程度（小學、國中....)
* eduyr: 教育年數
* hour: 上週實際工作小時（合計)
* earn_ad: 月薪
* stat1: 主要工作的工作身份(雇主、自營作業者、受政府雇用者、受私人雇用者、無酬家屬工作者)
* school: 在學狀況（在學中、畢業、異業、從未在正規學校求學過), 2007 年以後才有統計, 949, 727 missing values
keep year countyname sex age mar edu eduyr hour earn_ad stat1 cpi

* deal with the missing value
* type(countyname) = "str", others are double
* all 0 observations deleted
drop if countyname == ""
drop if year == . | sex == . | age == . | mar == .
drop if edu == . | eduyr == . | hour == . | earn_ad == . | stat1 == .

* slect the worker 
* dropped observations: 898,111 + 5,195
drop if earn_ad <= 0 | hour <= 0
* dropped observations: 43,130  + 150,902
drop if stat1 == 1 | stat1 == 2
drop stat1

* specify the year: 2000 - 2006
* 498,680(dropped) / 651,814(total)
drop if year < 2000 | year > 2006 
drop year

* recode sex (2=0)
* label define a2 2 "", modify
* label define a2 1 "男" 0 "女", replace

gen is_female  = "yes"
replace is_female = "no" if sex == 1
drop sex

* encode(countyname), generate(countycat)

* generate some additional variable
gen work_exp = age - (6 + eduyr)
label variable work_exp "working experience"

* gen martial = 0
* replace martial = 1 if mar == 2
* label variable martial "married"

* gen educat = 1
* replace educat = 2 if edu == 5 | edu == 6
* replace educat = 3 if edu > 6
* label variable educat "categorical varialbe for education"

gen is_lths = "no"
replace is_lths = "yes" if eduyr < 12
label variable is_lths "whether less than high school"

gen is_lths30 = "no"
replace is_lths30 = "yes" if eduyr < 12 & age < 30
label variable is_lths30 "wheter less than high school and age less than 30"

gen is_hsl = "no"
replace is_hsl = "yes" if eduyr <= 12
label variable is_hsl "wheter high school or less"

gen is_hsl30 = "no"
replace is_hsl30 = "yes" if eduyr <= 12 & age < 30
label variable is_hsl30 "wheter high school or less and age less than 30"


* egen agecat = cut(age), at(15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 95)
* label variable agecat "categorical variable for age"

gen is_teen = "no"
replace is_teen = "yes" if age < 20
label variable is_teen "wheter age less than 20"

* generate the response variable
gen wage = earn_ad / (hour * 4.2)
// gen real_wage = wage / (cpi / 100)

label variable wage "hourly wage"
drop hour earn_ad cpi

* treatment: wage < = 119
gen group1 = "treatment"
replace group1 = "control1" if (95*1.25) < wage
label variable group1 "response variable v1(2 category)"

gen group2 = "treatment"
replace group2 = "control1" if (95*1.25) < wage & wage <= 95*2
replace group2 = "control2" if 95*2 < wage
label variable group2 "response variable v2(3 category)"


save /Users/jackyyeh/github/minimum_wageML/data/processed/data_v2, replace
********************************************************************************

* create prediction data

********************************************************************************
// clear
// use $rdata/mu1991to2020.dta

