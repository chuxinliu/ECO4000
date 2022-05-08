# manually upload data file 

# install and load packages that are not preloaded in the environment
!pip install xlrd
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

# load data
# there is an issue with the xlsx file on the program, so we use this web data set provided by Pearson (which is the same)
df1 = pd.read_excel('cps08.xlsx')
# if an error occurs here, you probably want to open the excel file on excel, save again, and re-upload to Colab. This is due to some broken part of the original excel file.
df1.head()

# summary statistics
df1.describe()

# To estimate the constant term beta_0, we need to add a column of 1’s to our dataset
df1['const'] = 1
df1.head()

# regression 1: ahe on age
# Careful!!! There's a constant term in the exogenous variables!!!
reg1 = sm.OLS(endog=df1['ahe'], exog=df1[['age', 'const']])
results1 = reg1.fit()
print(results1.summary())

# regression 2: ahe on bachelor, female, age
reg2 = sm.OLS(endog=df1['ahe'], exog=df1[['bachelor', 'female', 'age', 'const']])
results2 = reg2.fit()
print(results2.summary())

# F test
regression = 'ahe~age+female+bachelor'
hypothesis = 'bachelor=female=0'
results_f1 =smf.ols(regression, df1).fit()
f_test1=results_f1.f_test(hypothesis)
print(f_test1)

# regression 3: ahe on female, age
reg3 = sm.OLS(endog=df1['ahe'], exog=df1[['female', 'age', 'const']])
results3 = reg3.fit()
print(results3.summary())

# F test
regression = 'ahe~age+female'
hypothesis = 'age=female=0'
results_f2 =smf.ols(regression, df1).fit()
f_test2=results_f2.f_test(hypothesis)
print(f_test2)

##################################################################

# generate log(ahe)
df1['lahe'] = np.log(df1['ahe'])
df1.head()

# regression 4
reg4 = sm.OLS(endog=df1['lahe'], exog=df1[['age','female', 'bachelor', 'const']])
results4 = reg4.fit()
print(results4.summary())

# generate log(age)
df1['lage'] = np.log(df1['age'])
df1.head()

# regression 5
reg5 = sm.OLS(endog=df1['lahe'], exog=df1[['lage','female', 'bachelor', 'const']])
results5 = reg5.fit()
print(results5.summary())

print(100*0.8039*(np.log(36)-np.log(35)))

# generate age-square
df1['age2'] = df1['age']*df1['age']
df1.head()

# regression 6
reg6 = sm.OLS(endog=df1['ahe'], exog=df1[['age', 'age2', 'female', 'bachelor', 'const']])
results6 = reg6.fit()
print(results6.summary())

# F test
regression = 'ahe~age+age2+female+bachelor'
hypothesis = 'age=age2=0'
results_f3 =smf.ols(regression, df1).fit()
f_test3=results_f3.f_test(hypothesis)
print(f_test3)

# generate age*female
df1['agefemale'] = df1['age']*df1['female']
df1.head()

# regression 7
reg7 = sm.OLS(endog=df1['lahe'], exog=df1[['age','agefemale', 'female', 'bachelor', 'const']])
results7 = reg7.fit()
print(results7.summary())

# generate age*bachelor
df1['agebachelor'] = df1['age']*df1['bachelor']
df1.head()

# regression 8
reg8 = sm.OLS(endog=df1['lahe'], exog=df1[['age','agebachelor', 'female', 'bachelor', 'const']])
results8 = reg8.fit()
print(results8.summary())