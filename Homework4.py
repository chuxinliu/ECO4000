# Here's the codes for Homework 4

!pip install linearmodels

!pip install xlrd

import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.iolib.summary2 import summary_col
from linearmodels.iv import IV2SLS

# Question 1 & 3: College Distance
# If you manually upload a csv file to binder, use the following code (but take the # out)
# df1 = pd.read_csv('CollegeDistance.csv')
# The next line is to use a dta file from an online source
df1 = pd.read_stata('https://wps.pearsoned.com/wps/media/objects/11422/11696965/empirical/empex_tb/CollegeDistance.dta')
df1.head()

df1.describe()

# To estimate the constant term beta_0, we need to add a column of 1’s to our dataset
df1['const'] = 1
df1.head()

# Run Regression!
# Careful!!! There's a constant term in the exogenous variables!!!
reg1 = sm.OLS(endog=df1['ed'], exog=df1[['dist','const']])
results = reg1.fit()
print(results.summary())

reg2 = sm.OLS(endog=df1['ed'], exog=df1[['dist', 'female', 'black', 'hispanic','bytest','dadcoll','incomehi','ownhome','cue80','stwmfg80','const']])
results = reg2.fit()
print(results.summary())

reg3 = sm.OLS(endog=df1['ed'], exog=df1[['dist', 'black', 'hispanic','const']])
results = reg3.fit()
print(results.summary())

regression = 'ed~dist+black+hispanic'
hypothesis = 'black=hispanic=0'
results =smf.ols(regression, df1).fit()
f_test=results.f_test(hypothesis)
print(f_test)

regression = 'ed~dist+female+black+hispanic+bytest+dadcoll+incomehi+ownhome+cue80+stwmfg80'
hypothesis = 'female=bytest=dadcoll=incomehi=ownhome=cue80=stwmfg80=0'
results =smf.ols(regression, df1).fit()
f_test=results.f_test(hypothesis)
print(f_test)

df1['cue80fraction'] = df1['cue80']/100
df1.head()

reg4 = sm.OLS(endog=df1['ed'], exog=df1[['dist', 'female', 'black', 'hispanic','bytest','dadcoll','incomehi','ownhome','cue80fraction','stwmfg80','const']])
results = reg4.fit()
print(results.summary())

#######################################################
#######################################################

# Question 2: CPS08
df2 = pd.read_stata('https://wps.pearsoned.com/wps/media/objects/11422/11696965/empirical/empex_tb/cps08.dta')
df2.head()

df2.describe()

# To estimate the constant term beta_0, we need to add a column of 1's to our dataset
df2['const'] = 1
df2.head()

reg5 = sm.OLS(endog=df2['ahe'], exog=df2[['age','const']])
results = reg5.fit()
print(results.summary())

reg6 = sm.OLS(endog=df2['ahe'], exog=df2[['age', 'bachelor', 'female','const']])
results = reg6.fit()
print(results.summary())

reg7 = sm.OLS(endog=df2['ahe'], exog=df2[['age', 'female', 'const']])
results = reg7.fit()
print(results.summary())

regression = 'ahe~age+female'
hypothesis = 'age=female=0'
results =smf.ols(regression, df2).fit()
f_test=results.f_test(hypothesis)
print(f_test)
