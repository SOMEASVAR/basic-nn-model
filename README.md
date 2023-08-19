# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

Explain the problem statement

## Neural Network Model

Include the neural network model diagram.

## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
```
Program developed by: Someasvar.R
Register Number: 212221230103
```
### Importing Modules:
```
from google.colab import auth
import gspread
from google.auth import default

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential as Seq
from tensorflow.keras.layers import Dense as Den

from tensorflow.keras.metrics import RootMeanSquaredError as rmse

import pandas as pd
import matplotlib.pyplot as plt
```
### Authenticate & Create Dataframe using Data in Sheets:
```
auth.authenticate_user()
creds, _ = default()
gc = gspread.authorize(creds)

sheet = gc.open('SomDocs DL-01').sheet1 
rows = sheet.get_all_values()

df = pd.DataFrame(rows[1:], columns=rows[0])
df = df.astype({'Input':'float'})
df = df.astype({'Output':'float'})
```
### Assign X and Y values:
```
x = df[["Input"]] .values
y = df[["Output"]].values
```
### Normalize the values & Split the data:
```
scaler = MinMaxScaler()
scaler.fit(x)
x_n = scaler.fit_transform(x)
x_train,x_test,y_train,y_test = train_test_split(x_n,y,test_size = 0.3,random_state = 3)
```
### Create a Neural Network & Train it:
```
ai_brain = Seq([
    Den(8,activation = 'relu',input_shape=[1]),
    Den(15,activation = 'relu'),
    Den(1),
])

ai_brain.compile(optimizer = 'rmsprop',loss = 'mse')

ai_brain.fit(x_train,y_train,epochs=2000)
ai_brain.fit(x_train,y_train,epochs=2000)
```
### Plot the Loss:
```
loss_plot = pd.DataFrame(ai_brain.history.history)
loss_plot.plot()
```
### Evaluate the model:
```
err = rmse()
preds = ai_brain.predict(x_test)
err(y_test,preds)
```
### Predict for some value:
```
x_n1 = [[9]]
x_n_n = scaler.transform(x_n1)
ai_brain.predict(x_n_n)
```

## Dataset Information

Include screenshot of the dataset

## OUTPUT
### Dataset values
![image](https://github.com/SOMEASVAR/basic-nn-model/assets/93434149/7ec73229-0815-45cc-a100-1e0cc9117b8d)



### Training Loss Vs Iteration Plot
![image](https://github.com/SOMEASVAR/basic-nn-model/assets/93434149/ea7f9c20-9ad7-4437-9470-7362aad0b08a)



### Test Data Root Mean Squared Error

![image](https://github.com/SOMEASVAR/basic-nn-model/assets/93434149/dc6ceb6d-feeb-4550-af31-1d4b66e2c779)


### New Sample Data Prediction

![image](https://github.com/SOMEASVAR/basic-nn-model/assets/93434149/88309ab7-c6e8-46c9-9222-40d25bef0822)


## RESULT
Thus to develop a neural network regression model for the dataset created is successfully executed.
