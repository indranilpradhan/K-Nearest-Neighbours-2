import numpy as np
import pandas as pd
from numpy.random import RandomState
from collections import Counter
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

class KNNClassifier:
    
    def __init__(self, k=2):
        self.k = k
        
    def fit(self, X_train, Y_train ):
        self.X_train = X_train
        self.Y_train = Y_train
        #self.X_validation = X_validation
        #self.Y_validation = Y_validation
        
    def fit_test(self, X_test):
        self.X_test = X_test
        
    def accuracy(self,y_real, y_pred):
#    print("y_real ",len(y_real))
#    print("y_pred ",len(y_pred))
        accuracy = np.sum(y_real == y_pred) / len(y_real)
        return accuracy
    
    def euclidean_distance(self,row):
    #print(row.shape)
        dist = []
        for train_row in self.X_train:
            dist.append(np.sqrt(np.sum((train_row - row) ** 2)))
        return dist
    
    def manhattan_distance(self,row):
        dist = []
        for train_row in self.X_train:
            dist.append(np.sum(np.abs(train_row - row)))
        return dist
    
    def prediction(self,row,k):
        dist = self.euclidean_distance(row)
        indexes = np.argsort(dist)[:self.k]
        neighbors = self.Y_train[indexes]
        match = Counter(neighbors).most_common(1)
        #print(match[0][0])
        return match[0][0]
    
    def prediction_manhattan(self,row,k):
        dist = self.manhattan_distance(row)
        indexes = np.argsort(dist)[:self.k]
        neighbors = self.Y_train[indexes]
        match = Counter(neighbors).most_common(1)
        #print(match[0][0])
        return match[0][0]
    
    def predict_knn(self,k):
        y_pred = [self.prediction(x,k) for x in self.X_test]
        return np.array(y_pred)
    
    def predict_euclidean(self,k):
        y_pred = [self.prediction(x,k) for x in self.X_test]
        return np.array(y_pred)
    
    def predict_manhattan(self,k):
        y_pred = [self.prediction_manhattan(x,k) for x in self.X_test]
        return np.array(y_pred)
    
    def train(self,path):
        missing_values = ["?"]
        temp_df = pd.read_csv(str(path), na_values = missing_values, header = None)
        df = temp_df.fillna(temp_df.iloc[:,11:12].mode().iloc[0])
        #rng = RandomState()
        #temp_train = df.sample(frac=0.8,random_state = rng)
        #temp_validation = df.loc[~df.index.isin(temp_train.index)]
        train = np.array(df)
        #validation = np.array(temp_validation)
        for x in range(0, train.shape[0]):
            for y in range(0, train.shape[1]):
                train[x][y] = float(ord(train[x][y]))
        # for x in range(0, validation.shape[0]):
        #     for y in range(0, validation.shape[1]):
        #         validation[x][y] = float(ord(validation[x][y]))
        X_train,Y_train = train[:,1:], train[:,0]
        #X_validation,Y_validation = validation[:, 1:], validation[:,0]
        self.fit(X_train,Y_train)
        #X_train = np.array(X_train)
        #Y_train = np.array(Y_train)
        #X_validation = np.array(X_validation)
        #Y_validation = np.array(Y_validation)
        #X_train.shape
        
    def predict(self,path):
        df_test = pd.read_csv(str(path),header = None)
        #print(df_test.shape)
        X_test = df_test.to_numpy()
        for x in range(0, X_test.shape[0]):
            for y in range(0, X_test.shape[1]):
                X_test[x][y] = float(ord(X_test[x][y]))
        self.fit_test(X_test)
        #print(X_test)
        # Y_temp_test = pd.read_csv('/media/indranil/New Volume/second sem/SMAI/Assignment 1/q2/dataset/test_labels.csv',header = None)
        # Y_test_np = Y_temp_test.to_numpy()
        # #Y_test.shape
        # Y_list = list()
        # for x in range(0, Y_test_np.shape[0]):
        #     for y in range(0, Y_test_np.shape[1]):
        #         #print(float(ord(Y_test_np[x][y])))
        #         Y_list.append(float(ord(Y_test_np[x][y])))
        # Y_test = np.array(Y_list)
        predictions_k = self.predict_euclidean(2)
        res = list()
        for i in predictions_k:
            res.append(chr(int(i)))
        return res
        #accuracy(Y_test, predictions_k)