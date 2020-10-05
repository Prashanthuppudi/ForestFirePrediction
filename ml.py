import pickle
import warnings

import numpy as np
import pandas as pd
from sklearn import svm

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix



warnings.filterwarnings("ignore")

dataframe =pd.read_csv("forest3.csv")
#a=corr_matrix=dataframe.corr()
#print(a)
#b=corr_matrix["area"].sort_values(ascending=False)
#print(b)
data = np.array(dataframe)

x = data[0:, 4:-1]
y = data[0:, -1]
y = y.astype('float')
x = x.astype('float')

#print(x)
#print(y)

x_train, x_test, y_train, y_test = train_test_split(x, y,test_size=0.3, random_state=10)


#print("svm")

svm_model =svm.SVC(probability=True,kernel="linear",C=1)
svm_model.fit(x_train,y_train)






pickle.dump(svm_model, open("model.pkl", "wb"))
