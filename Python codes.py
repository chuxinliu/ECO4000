# Sargent's Codes

!pip install linearmodels

!pip install xlrd

import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import pandas as pd
import statsmodels.api as sm
from statsmodels.iolib.summary2 import summary_col
from linearmodels.iv import IV2SLS

df1 = pd.read_excel('Growth.xls')
df1.head()

df1.describe()

# Scatter Plot!
plt.style.use('seaborn')
df1.plot(x='tradeshare', y='growth', kind='scatter')
plt.show()

# To estimate the constant term beta_0, we need to add a column of 1’s to our dataset
df1['const'] = 1
df1.head()

# Run Regression!
# Careful!!! There's a constant term in the exogenous variables!!!
reg1 = sm.OLS(endog=df1['growth'], exog=df1[['tradeshare', 'const']])
results = reg1.fit()
print(results.summary())

## obtain an array of predicted growth for every value of tradeshare
df1_plot = df1.dropna(subset=['growth', 'tradeshare'])
fix, ax = plt.subplots()
ax.scatter(df1_plot['tradeshare'], results.predict(), alpha=0.5, label='predicted')
## Plot observed values and predicted values, add the regression line
ax.scatter(df1_plot['tradeshare'], df1_plot['growth'], alpha=0.5, label='observed')
ax.legend()
ax.set_title('OLS predicted values')
ax.set_xlabel('tradeshare')
ax.set_ylabel('growth')
X=df1['tradeshare']
y_pred = results.predict(X)
plt.plot(X, y_pred, color='blue', linewidth=2)
plt.show()


# Omitted Variable? Multiple Regression!
# Careful!!! There's a constant term in the exogenous variables!!!
reg2 = sm.OLS(endog=df1['growth'], exog=df1[['tradeshare', 'rgdp60', 'yearsschool', 'rev_coups', 'const']])
results = reg2.fit()
print(results.summary())
