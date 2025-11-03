from sklearn.datasets import fetch_california_housing
from  sklearn.model_selection import  train_test_split
from regressionnderidge import ridgeression
#import joblib as jb
import matplotlib.pyplot as plt

housing=fetch_california_housing(as_frame=True)
x=housing.data
y=housing.target
df=housing.frame
X_train_df,X_test_df,y_train,y_test=train_test_split(x,y,test_size=0.3)
X_train_df = X_train_df.values
X_test_df  = X_test_df.values
y_train = y_train.values
y_test = y_test.values
plt.scatter(X_test_df[:,5],X_test_df[:,2])
plt.show()




Model2=ridgeression(lambd=0)
X_train=Model2.ajout_du_biais(X_train_df,condition_de_biais=True)
theta_final = Model2.fit(X_train_df,y_train)
Perf2=Model2.score(X_train_df, y_train)
Perf=Model2.score(X_test_df, y_test)
print(theta_final,Perf,Perf2)





