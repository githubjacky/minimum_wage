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
* type(countyname) = "str"
drop if countyname == ""  // type: str \\
    year == . | sex == . | age == . | mar == .  // type: double
    edu == . | eduyr == . | hour == . | earn_ad == . | stat1 == .

* slect the worker 
drop if earn_ad <= 0 | hour <= 0 // dropped observations: 898,111 + 5,195
drop if stat1 == 1 | stat1 == 2  // dropped observations: 43,130  + 150,902
drop stat1

* specify the year: 2000 - 2006
drop if year < 2000 | year > 2006 // 498,680(dropped) / 651,814(total)
drop year

recode sex (2=0)
label define a2 2 "", modify
label define a2 1 "男" 0 "女", replace

encode(countyname), generate(countycat)

* generate some additional variable
gen work_exp = age - (6 + eduyr)
label variable work_exp "working experience"

gen martial = 0
replace martial = 1 if mar == 2
label variable martial "married"

gen educat = 1
replace educat = 2 if edu == 5 | edu == 6
replace educat = 3 if edu > 6
label variable educat "categorical varialbe for education"

gen lths = 0
replace lths = 1 if eduyr < 12
label variable lths "less than high school"

gen lths30 = 0
replace lths30 = 1 if eduyr < 12 & age < 30
label variable lths30 "less than high school and age less than 30"

gen hsl = 0
replace hsl = 1 if eduyr <= 12
label variable hsl "high school or less"

gen hsl30 = 0
replace hsl30 = 1 if eduyr <= 12 & age < 30
label variable hsl30 "high school or less and age less than 30"


egen agecat = cut(age), at(15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 95)
label variable agecat "categorical variable for age"

gen teen = 0
replace teen = 1 if age < 20

* generate the response variable
gen wage = earn_ad / (hour * 4.2)
// gen real_wage = wage / (cpi / 100)

label variable wage "hourly wage"
drop hour earn_ad cpi

gen group = 1  // <= 119
replace group = 2 if (95*1.25) < wage & wage <= 95*2
replace group = 3 if 95*2 < wage


save $wdata/training.dta, replace


********************************************************************************

* create prediction data

********************************************************************************
// clear
// use $rdata/mu1991to2020.dta

