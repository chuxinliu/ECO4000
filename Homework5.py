# Here's the codes for q3 and q4 in Homework 5

!pip install linearmodels

import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.iolib.summary2 import summary_col
from linearmodels.iv import IV2SLS

# Question 3: CPS80
df1 = pd.read_stata('https://wps.pearsoned.com/wps/media/objects/11422/11696965/empirical/empex_tb/cps08.dta')
df1.head()

df1.describe()

# To estimate the constant term beta_0, we need to add a column of 1's to our dataset
df1['const'] = 1
df1.head()

reg1 = sm.OLS(endog=df1['ahe'], exog=df1[['age','female', 'bachelor', 'const']])
results = reg1.fit()
print(results.summary())

# generate log(ahe)
df1['lahe'] = np.log(df1['ahe'])
df1.head()

reg2 = sm.OLS(endog=df1['lahe'], exog=df1[['age','female', 'bachelor', 'const']])
results = reg2.fit()
print(results.summary())

# generate log(age)
df1['lage'] = np.log(df1['age'])
df1.head()

reg3 = sm.OLS(endog=df1['lahe'], exog=df1[['lage','female', 'bachelor', 'const']])
results = reg3.fit()
print(results.summary())

print(100*0.8039*(np.log(36)-np.log(35)))

# generate age-square
df1['age2'] = df1['age']*df1['age']
df1.head()

reg4 = sm.OLS(endog=df1['ahe'], exog=df1[['age', 'age2', 'female', 'bachelor', 'const']])
results = reg4.fit()
print(results.summary())

regression = 'ahe~age+age2+female+bachelor'
hypothesis = 'age=age2=0'
results =smf.ols(regression, df1).fit()
f_test=results.f_test(hypothesis)
print(f_test)

# generate age*female
df1['agefemale'] = df1['age']*df1['female']
df1.head()

reg5 = sm.OLS(endog=df1['lahe'], exog=df1[['age','agefemale', 'female', 'bachelor', 'const']])
results = reg5.fit()
print(results.summary())

# generate age*bachelor
df1['agebachelor'] = df1['age']*df1['bachelor']
df1.head()

reg6 = sm.OLS(endog=df1['lahe'], exog=df1[['age','agebachelor', 'female', 'bachelor', 'const']])
results = reg6.fit()
print(results.summary())

#############################################################
#############################################################

# Question 4: College Distance
# If you manually upload a csv file to binder, use the following code (but take the # out)
# df2 = pd.read_csv('CollegeDistance.csv')
# The next line is to use a dta file from an online source
df2 = pd.read_stata('https://wps.pearsoned.com/wps/media/objects/11422/11696965/empirical/empex_tb/CollegeDistance.dta')
df2.head()

df2.describe()

# To estimate the constant term beta_0, we need to add a column of 1’s to our dataset
df2['const'] = 1
df2.head()

# Run Regression!
# Careful!!! There's a constant term in the exogenous variables!!!
reg7 = sm.OLS(endog=df2['ed'], exog=df2[['dist', 'female', 'black', 'hispanic','bytest', 'tuition','dadcoll','momcoll','incomehi','ownhome','cue80','stwmfg80','const']])
results = reg7.fit()
print(results.summary())

# generate log(ahe)
df2['led'] = np.log(df2['ed'])
df2.head()

reg8 = sm.OLS(endog=df2['led'], exog=df2[['dist', 'female', 'black', 'hispanic','bytest', 'tuition','dadcoll','momcoll','incomehi','ownhome','cue80','stwmfg80','const']])
results = reg8.fit()
print(results.summary())

# generate dist-square
df2['dist2'] = df2['dist']*df2['dist']
df2.head()

reg9 = sm.OLS(endog=df2['ed'], exog=df2[['dist','dist2', 'female', 'black', 'hispanic','bytest', 'tuition','dadcoll','momcoll','incomehi','ownhome','cue80','stwmfg80','const']])
results = reg9.fit()
print(results.summary())

# generate incomehi*tuition
df2['incomehit'] = df2['incomehi']*df2['tuition']
df2.head()

reg10 = sm.OLS(endog=df2['ed'], exog=df2[['dist', 'female', 'black', 'hispanic','bytest', 'tuition','incomehi','incomehit','dadcoll','momcoll','ownhome','cue80','stwmfg80','const']])
results = reg10.fit()
print(results.summary())