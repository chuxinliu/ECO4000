#install packages that are not preloaded in the environment
!pip install statsmodels

!pip install xlrd

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

# load data
# there is an issue with the xlsx file on the program, so we use this web data set provided by Pearson (which is the same)
df1 = pd.read_stata('https://wps.pearsoned.com/wps/media/objects/11422/11696965/empirical/empex_tb/cps08.dta')
df1.head()

# summary statistics
df1.describe()

# To estimate the constant term beta_0, we need to add a column of 1’s to our dataset
df1['const'] = 1
df1.head()

# regression 1: ahe on age
# Careful!!! There's a constant term in the exogenous variables!!!
reg1 = sm.OLS(endog=df1['ahe'], exog=df1[['age', 'const']])
results = reg1.fit()
print(results.summary())

# regression 2: ahe on bachelor, female, age
reg2 = sm.OLS(endog=df1['ahe'], exog=df1[['bachelor', 'female', 'age', 'const']])
results = reg2.fit()
print(results.summary())

# F test
regression = 'ahe~age+female+bachelor'
hypothesis = 'bachelor=female=0'
results =smf.ols(regression, df1).fit()
f_test=results.f_test(hypothesis)
print(f_test)

# regression 3: ahe on female, age
reg3 = sm.OLS(endog=df1['ahe'], exog=df1[['female', 'age', 'const']])
results = reg3.fit()
print(results.summary())

# F test
regression = 'ahe~age+female'
hypothesis = 'age=female=0'
results =smf.ols(regression, df1).fit()
f_test=results.f_test(hypothesis)
print(f_test)

##################################################################

# generate log(ahe)
df1['lahe'] = np.log(df1['ahe'])
df1.head()

# regression 4
reg4 = sm.OLS(endog=df1['lahe'], exog=df1[['age','female', 'bachelor', 'const']])
results = reg4.fit()
print(results.summary())

# generate log(age)
df1['lage'] = np.log(df1['age'])
df1.head()

# regression 5
reg5 = sm.OLS(endog=df1['lahe'], exog=df1[['lage','female', 'bachelor', 'const']])
results = reg5.fit()
print(results.summary())

print(100*0.8039*(np.log(36)-np.log(35)))

# generate age-square
df1['age2'] = df1['age']*df1['age']
df1.head()

# regression 6
reg6 = sm.OLS(endog=df1['ahe'], exog=df1[['age', 'age2', 'female', 'bachelor', 'const']])
results = reg6.fit()
print(results.summary())

# F test
regression = 'ahe~age+age2+female+bachelor'
hypothesis = 'age=age2=0'
results =smf.ols(regression, df1).fit()
f_test=results.f_test(hypothesis)
print(f_test)

# generate age*female
df1['agefemale'] = df1['age']*df1['female']
df1.head()

# regression 7
reg7 = sm.OLS(endog=df1['lahe'], exog=df1[['age','agefemale', 'female', 'bachelor', 'const']])
results = reg7.fit()
print(results.summary())

# generate age*bachelor
df1['agebachelor'] = df1['age']*df1['bachelor']
df1.head()

# regression 8
reg8 = sm.OLS(endog=df1['lahe'], exog=df1[['age','agebachelor', 'female', 'bachelor', 'const']])
results = reg8.fit()
print(results.summary())
