import numpy as np
import pandas as pd
import os
os.chdir(r"C:\Users\sonim\OneDrive\Desktop\New folder\Python\Class 6 ML linear Regression\Linear regression\PRACTICE FILES")

## Part 1 - Data Preprocessing
### Importing the dataset
pd.set_option('display.max_columns', None)

dataset = pd.read_csv('car_purchasing.csv', encoding="latin1")
dataset.head()

dataset = dataset.drop(["customer name","JobTitle","customer e-mail","country","TotalPayBenefits","Benefits","BasePay","OvertimePay","OtherPay"],axis = 1)
dataset.head()

dataset.describe()

import matplotlib.pyplot as plt
import statsmodels.formula.api as sm
# boxplot to showoutliers
dataset.boxplot(column=["age"])
plt.show()

dataset.boxplot(column=["TotalPay"])
plt.show()

dataset.boxplot(column=["credit card debt"])
plt.show()

dataset.boxplot(column=["net worth"])
plt.show() 

dataset.boxplot(column=["car purchase amount"])
plt.show()

# easy way to remove outliers
def pintu (data,age):
 Q1 = data[age].quantile(0.25)
 Q3 = data[age].quantile(0.75)
 IQR = Q3 - Q1
 data= data.loc[~((data[age] < (Q1 - 1.5 * IQR)) | (data[age] > (Q3 + 1.5 * IQR))),]
 return data

dataset = pintu(dataset,"age")
dataset.boxplot(column=["age"])

dataset = pintu(dataset,"TotalPay")
dataset.boxplot(column=["TotalPay"])

dataset = pintu(dataset,"credit card debt")
dataset.boxplot(column=["credit card debt"])

dataset = pintu(dataset,"net worth")
dataset.boxplot(column=["net worth"])

dataset = pintu(dataset,"car purchase amount")
dataset.boxplot(column=["car purchase amount"])

#replacing space with '_' in column names
dataset.columns = dataset.columns.str.replace(' ', '_')
print(dataset.columns)


rock=sm.ols(formula=
"car_purchase_amount~age+net_worth",
data=dataset).fit()
rock.summary()# shows total summary
#Prob (F-statistic) is the ANOVA; should be less than 0.05

#predicting the outcomes
dataset["pred"] = rock.predict()
dataset.head()


var = pd.DataFrame(round(rock.pvalues,3))# shows p value
rock.rsquared
var["coeff"] = rock.params#coefficients

from statsmodels.stats.outliers_influence import variance_inflation_factor
variables = rock.model.exog #.if I had saved data as rock
# this it would have looked like rock.model.exog
vif = [variance_inflation_factor(variables, i) for i in range(variables.shape[1])]
vif 
var["vif"] = vif
var

###### mape
dataset["mp"] = abs((dataset["car_purchase_amount"] - dataset["pred"])/dataset["car_purchase_amount"])
(dataset.mp.mean())*100#mape

