# step-1 : load raw data
# step-2 : Identifying  x and y (input or output)
# step-3 : split data into train and test

import pandas as pd
from sklearn.model_selection import train_test_split


def load_and_split_data():
    # step-1 : load raw data
    data=pd.read_csv("../data/raw/Insurances_data.csv")

    # step-2 : Identifying  x and y (input or output)
    x=data[['Age','Annual_Income_LPA','Policy_Term_Years','Sum_Assured_Lakhs']]
    y=data['Annual_Premium_Thousands']

    # step-3 : split data into train and test
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

    return x_train,x_test,y_train,y_test

