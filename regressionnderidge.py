import numpy as np
import joblib

class ridgeression:
    def __init__(self,condition_de_biais=True,lambd=None):
        self.theta=None
        self.biais=None
        if condition_de_biais is None:
           condition_de_biais = True  # valeur par défaut conditionnelle
        self.condition_de_biais=condition_de_biais
        self.lambd=lambd
       
        
        
    def ajout_du_biais(self, X, condition_de_biais=None):
    # Si aucune condition n’est passée, on utilise celle de l’objet
        if condition_de_biais is None:
            condition_de_biais = self.condition_de_biais

        if condition_de_biais:
            X = np.hstack((np.ones((X.shape[0], 1)), X))
    
        return X

        
        
    def fit(self,X,Y):
        X_b=self.ajout_du_biais(X,condition_de_biais=None)
        # Matrice de regularisation 
        I=np.eye(X_b.shape[1])
        
        if self.condition_de_biais:
            I[0,0]=0 # on ne penalise pas le biais
        A= X_b.T @ X_b + self.lambd * I
        b = X_b.T @ Y
        theta= np.linalg.inv(A) @ b
        
        if self.condition_de_biais:
            self.biais= theta[0]
            self.theta=theta[1:]
        else:
            self.biais=0
            self.theta= theta
        return self.theta
    
    
    def predict(self, X):
        """Prédit les valeurs de y pour de nouvelles données"""
        if self.theta is None:
            raise ValueError("Le modèle n’a pas encore été entraîné. Utilise .fit() d’abord.")
        X_b = self.ajout_du_biais(X, condition_de_biais=None)
        if self.condition_de_biais:
            
            theta= np.concatenate([[self.biais], self.theta]) 
        else:
            self.theta=theta
        return X_b @ theta
    
    
    def score(self, X, Y):
        """Renvoie le R² (coefficient de détermination)"""
        y_pred = self.predict(X)
        u = np.sum((Y - y_pred) ** 2)
        v = np.sum((Y - np.mean(Y)) ** 2)
        return 1 - u / v
    


Model2=ridgeression()
joblib.dump(Model2,filename="model_california2.joblib",compress=0)                
                
            