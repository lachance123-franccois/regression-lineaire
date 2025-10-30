#from sklearn.datasets import make_regression
import numpy as np
import joblib
import matplotlib.pyplot as plt


class MonModele:
    def __init__(self):
        self.theta = None  # sera calculé plus tard 
        self.X=None
        self.y=None
       
   # def fonction(self,x):
     #   X = np.hstack((np.ones((x.shape[0], 1)), x))

        
#        X=np.hstack((x,x,x,x,x,X))
       # return X
    
        
    def model(self,X,theta):
        return np.dot(X,theta)
   
    def gradient(self,X,y):
        return -2*X.T.dot(y)+2*X.T.dot(self.X)*self.theta
    def moindre_carre(self,X,y):
        XT=X.T
        A1=np.dot(XT,X)
        A=np.linalg.inv(A1)
        B=np.dot(XT,y)
        self.theta=np.dot(A,B)
        return self.theta
    
    def performance (self,y,preds):
        u=np.sum(((y-preds)**2))
        v=np.sum((y-y.mean())**2)
        return 1-(u/v)
    def process_dataframe(df):
    
        #print("performance",performance(y,predictions))
       # plt.scatter(x,y)
        #plt.scatter(x,predictions,color='r')
        plt.suptitle('donnees et modeles')
        plt.show()
        
        
Model=MonModele()


joblib.dump(Model,filename="model_california.joblib",compress=0)

    




