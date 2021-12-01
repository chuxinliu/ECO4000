
#install packages that are not preloaded in the environment

!pip install statsmodels

!pip install xlrd


import numpy as np
import pandas as pd
import statsmodels.api as sm

# load data
df1 = pd.read_excel('Growth.xls')
df1.head()

# summary statistics
df1.describe()

# To estimate the constant term beta_0, we need to add a column of 1’s to our dataset
df1['const'] = 1
df1.head()

# Run Regression!
# Careful!!! There's a constant term in the exogenous variables!!!
reg1 = sm.OLS(endog=df1['growth'], exog=df1[['tradeshare', 'const']])
results = reg1.fit()
print(results.summary())

# Omitted Variable? Multiple Regression!
# Careful!!! There's a constant term in the exogenous variables!!!
reg2 = sm.OLS(endog=df1['growth'], exog=df1[['tradeshare','yearsschool','rgdp60','rev_coups','assasinations','const']])
results = reg2.fit()
print(results.summary())

# drop the 64th row, which is Malta
df2 = df1
df2 = df2.drop(index=[64])
df2

# same regression, using df2 (without Malta)
reg3 = sm.OLS(endog=df2['growth'], exog=df2[['tradeshare','yearsschool','rgdp60','rev_coups','assasinations','const']])
results = reg3.fit()
print(results.summary())

# back to df1, add a variable "oil"
reg4 = sm.OLS(endog=df1['growth'], exog=df1[['oil', 'tradeshare','yearsschool','rgdp60','rev_coups','assasinations','const']])
results = reg4.fit()
print(results.summary())
