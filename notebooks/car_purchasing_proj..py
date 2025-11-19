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

import numpy as np
import pandas as pd
import tensorflow as tf
tf.__version__
import os
os.chdir("C:\\Users\\ASUS\\Desktop\\Python\\Class 11 Deep Learning etc\\DL\\regression")
## Part 1 - Data Preprocessing
### Importing the dataset
import openpyxl
dataset = pd.read_excel('reg_data.xlsx',engine = 'openpyxl')

X = dataset.drop("PE",axis=1).values
y = dataset.loc[:, "PE"].values

### Splitting the dataset into the Training set and Test set"""
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


## Part 2 - Building the ANN
### Initializing the ANN
ann = tf.keras.models.Sequential()

### Adding the input layer and the first hidden layer"""
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))

### Adding the second hidden layer"""
# don't have to specify no. of input var, 
#since python will automatically detect that

ann.add(tf.keras.layers.Dense(units=6, activation='relu'))

### Adding the third hidden layer, 
#copy paste the previous one. (here 2nd layer not required)
#ann.add(tf.keras.layers.Dense(units=6, activation='relu'))


### Adding the output layer"""
#only 1 output hence 1 layer
ann.add(tf.keras.layers.Dense(units=1))

## Part 3 - Training the ANN

### Compiling the ANN
ann.compile(optimizer = 'adam', loss = 'mean_squared_error')

### Training the ANN model on the Training set"""
ann.fit(X_train, y_train, batch_size = 32, epochs = 100)

### Predicting the results of the Test set"""

y_pred = ann.predict(X_test)


# Calculate MAPE
y_pred = pd.DataFrame(y_pred)
y_test = pd.DataFrame(y_test)

mp = abs((y_test - y_pred)/y_test)
(mp.mean())*100#mape
#0.87%

#data["mp"] = abs((data["Price_house"] - data["pred"])/data["Price_house"])
#(data.mp.mean())*100#mape

















































X = dataset.drop("car purchase amount",axis=1).values
y = dataset.loc[:, "car purchase amount"].values

dataset.isnull().sum()

### Splitting the dataset into the Training set and Test set"""
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# Part 2 - Now let's make the ANN!
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import backend

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
#units:- number of neurons that will be present 
#in the respective layer
#units = (total ind var + dep var)/2 - (it's a tip)
# input_dim = num of ind var
# initialize the weights with the function "uniform"
# rectifier funciton for input layer & sigmoid for output 
# together it's called relu(rectified linear unit)

classifier.add(Dense(units=6, 
kernel_initializer='uniform', activation='relu',
input_dim=9))

# Adding the second hidden layer
# just copy paste the above line without input_dim
classifier.add(Dense(units=6, 
kernel_initializer='uniform', activation='relu'))

# Adding the output layer
#output is binary, hence units will be 1; 
#sigmoid will give us probabilities (dep var binary)
# dep var as nominal use softmax as activation function
classifier.add(Dense(units=1, 
kernel_initializer='uniform', activation='sigmoid'))


# Compiling the ANN
# initialized weights should be optimized using "adam"
#function. This uses stochastic gradient descent.
# loss function for binary is 'binary_crossentropy'
# metrics to be evaluated
# optimizer  could be adam or rmsprop
classifier.compile(optimizer='adam', 
loss='binary_crossentropy', metrics=['accuracy'])

# Fitting the ANN to the Training set
#batch size is no. of observation model will adjust weights
# thumb rule batch size 10 and epochs=100
#
classifier.fit(X_train, y_train, batch_size=10,
epochs=100, validation_split=0.1)

# Part 3 - Making predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = np.where(y_pred > 0.5,1,0)

# Making the Confusion Matrix
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print(accuracy_score(y_test, y_pred))


from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
##Computing false and true positive rates
fpr, tpr,_=roc_curve(y_pred,y_test,drop_intermediate=False)

import matplotlib.pyplot as plt
plt.figure()
##Adding the ROC
plt.plot(fpr, tpr, color='red',
 lw=2, label='ROC curve')
##Random FPR and TPR
plt.plot([0, 1], [0, 1], color='blue', lw=2, linestyle='--')
##Title and label
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC curve')
plt.show()
roc_auc_score(y_pred,y_test)
#AUC value
