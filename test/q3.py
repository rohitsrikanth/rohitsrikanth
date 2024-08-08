import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
data=load_iris()
x,y=data.data,data.target
x_1,x_2,y_1,y_2=train_test_split(x,y,test_size=0.2)
model=LogisticRegression(random_state=42)
model.fit(x_1,y_1)
y_pred=model.predict(x_2)
acc=accuracy_score(y_2,y_pred)
con=confusion_matrix(y_2,y_pred)
print(f"accuracy:{acc}\nconfusion matrix:{con}")
