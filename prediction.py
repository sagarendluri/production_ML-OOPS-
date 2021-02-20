import joblib
import pymysql
import json
import sqlalchemy
import pandas as pd
from smart_open import smart_open 
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from DB import localdb
class DataPrediction():
    def __init__(self,data,i,path):
        self.data = data
        self.i = i
        self.path = path
    def split(self):
        X = self.data.drop([self.i,'Unnamed: 0.1'],axis=1)
        y = self.data[self.i] 
        X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2,random_state = 42)
        joblib_file = (smart_open(self.path))
        joblib_LR_model = joblib.load(joblib_file)
        score = joblib_LR_model.score(X_train,y_train)
        print("Test score: {0:.2f} %".format(100 * score))
        y_pred = joblib_LR_model.predict(X_test)
        cm =confusion_matrix(y_test, y_pred)
        Accuracy = metrics.accuracy_score(y_test, y_pred)
        cr = {"performance":{"Algorithm": self.path,
                          "performance_matrix": 'performance_matrix',
                          },
          "Accuracy":Accuracy,
          "confusion matrix": cm,
          "grid": 'predicted_model',
          "Estimator": 'Predicted_model',
          "model_file_name": 'loaded',
          "request_payload_data": 'loaded',
          "dataset_name":'loaded'
          }
        mySql_insert_query = """INSERT INTO Phenotype_model_info(model_file_path,
                                            confusion_matrix, hyperparameter_grid, best_estimator,
                                            dataset_name, request_payload,Phenotypes_id_id, performence)
                                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s) """
        request_payload_data= 'loaded'
        esti = {'estimator':str('Predicted_model')}
        cm = {'confusion_metrics':cm.tolist()}
        cr = {'Total_perfomence':str(cr)}
        grid = {'Best_grid':str('predicted_model')}
        model= self.path
        dname = self.path
        l=1
        cursor ,conn = localdb()
        recordTuple = (model,
                            json.dumps(cm), 
                            json.dumps(grid),
                            json.dumps(esti), 
                            dname, json.dumps(request_payload_data),
                            l,json.dumps(cr)
            )
        cursor.execute(mySql_insert_query, recordTuple)
        conn.commit()
        print('Loaded')
