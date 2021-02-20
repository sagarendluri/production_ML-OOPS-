import pandas as pd
from configparser import ConfigParser
import numpy as np
from sklearn.preprocessing import LabelEncoder, Normalizer, StandardScaler
from sklearn import preprocessing
class DataCleaning(object):
    def __init__(self,df1,i,ini):
        self.df1 = df1
        self.i = i
        self.ini=  ini
        self.config_object = ConfigParser()
        self.config_object.read(self.ini)
        drop_cols = ['Unnamed: 0','alleles','chrom','pos', 'strand','assembly#','center','protLSID',
                   'assayLSID','panelLSID','QCcode']
        drop_cols.append(i)
        cols = df1.columns.intersection(drop_cols)
        self.config_object.read(self.ini)
        self.userinfo = self.config_object["MYSQL"]
        self.password= self.userinfo["password"]
        self.user= self.userinfo["user"]
        self.host= self.userinfo["host"]
        self.db= self.userinfo["db"]
        self.access_key_id = self.userinfo["access_key_id"]
        self.secret_access_key  = self.userinfo["secret_access_key"]
        self.bucket  = self.userinfo["bucket"]
        self.x = df1.drop(cols,axis = 1)
        self.y = df1[self.i]
    def dtypes_handliing(self):
        try:
            print("Data analyzing")
            # Select the bool columns
            self.bool_col = self.x.select_dtypes(include='bool')
            # select the float columns
            self.float_col = self.x.select_dtypes(include=[np.float64])
            # select the int columns
            self.int_col = self.x.select_dtypes(include=[np.int64])
        #   # select non-numeric columns
            self.cat_col = self.x.select_dtypes(include=['category', object])
            self.date_col = self.x.select_dtypes(include=['datetime64'])
            return self.bool_col,self.float_col,self.int_col,self.cat_col,self.date_col
        except:
            print('Data analyzing failed')
    def handiling_categorical(self):
        try:
            self.cat_result = []
            df_most_common_imputed = self.cat_col.apply(lambda x: x.fillna(x.value_counts().index[0]))
            print(df_most_common_imputed)
            for col in list(df_most_common_imputed):
                labels, levels = pd.factorize(self.cat_col[col].unique())
                if sum(labels) <= self.cat_col[col].shape[0]:
                    self.cat_result.append(pd.get_dummies(self.cat_col[col], prefix=col))
            print("Categorical Data analyzing")
            return self.cat_result
        except:
            print("Categorical Data analyzing failed")
    def handiling_int_col(self):
        try:
            print("int Data analyzing")
            self.int_result = []
            for col in self.int_col:
                labels, levels = pd.factorize(self.int_col[col].unique())
                if len(labels) == self.int_col[col].shape[0]:
                    re = self.int_col.drop([col],axis=1)
                else:
                    self.int_result.append(self.int_col[col])
            return self.int_result
        except:
            print("int Data analyzing failed")
    def concat_cat(self):
        result = [self.cat_result]        
        for fname in result:
            if fname == []:
                print('No objects to concat')
            else:
                self.data =  pd.concat([col for col in fname],axis=1)
                self.cleaned_Data_frm = pd.concat([self.data.reindex(self.y.index)], axis=1)
                print(list(self.cleaned_Data_frm.columns))
                return self.cleaned_Data_frm
    def concat_int(self):
        result2 = [self.int_result]
        for fname2 in result2:
            if fname2 == []:
                print('No int_cols to concat')
            else:
                self.data2 =  pd.concat([col for col in fname2],axis=1)
                self.cleaned_Data_frm1 = pd.concat([self.data2.reindex(self.y.index)], axis=1)
                return self.cleaned_Data_frm1
    def encoder(self):
        if(self.y.dtype == object or self.y.dtype == bool):
            label_encoder = preprocessing.LabelEncoder()
            self.y= label_encoder.fit_transform(self.y.astype(str))
            self.dataset = pd.DataFrame()
            self.dataset[self.i] = self.y.tolist()
            print('Multiclass_classification')
            self.types = 'Classification_problem'
            return self.dataset[self.i]
        elif self.y.dtypes == np.int:
            self.types = 'Classification_problem'
            print('Classification_problem')
            return self.y
        else:
            print('Regression_problem')
            return self.y
    def QC(self,cleaned_Data_frm, cleaned_Data_frm1,y):
        try:
            print('Models Building')
            float_cols = self.float_col
            result = pd.concat([cleaned_Data_frm,cleaned_Data_frm1,y,float_cols], axis=1)
            self.data_sorted1 = result.sort_values(self.i)
            data_sorted = self.data_sorted1.loc[:,~self.data_sorted1.columns.duplicated()]
            print(data_sorted.shape)
            return data_sorted
        except:
            print('data returned faild')
    def classification(self,cleaned_Data_frm1, cleaned_Data_frm,y):
        try:
            print("Model building")
            float_cols = self.float_col
            result = pd.concat([cleaned_Data_frm1,cleaned_Data_frm,y,float_cols], axis=1)
            self.data_sorted1 = result.loc[:,~result.columns.duplicated()]
            self.data_sorted2 =  self.data_sorted1.sort_values(self.i)
            self.data_sorted = self.data_sorted2.dropna(thresh=self.data_sorted2.shape[0]*0.5,how='all',axis=1)
            data_sorted  = self.data_sorted.dropna()
            return data_sorted
        except:
            print('data returned faild')
def train_test(X_train,X_test):
    try:
        vs_constant = VarianceThreshold(threshold=0)
        # select the numerical columns only.
        numerical_x_train = X_train[X_train.select_dtypes([np.number]).columns]
        # fit the object to our data.
        vs_constant.fit(numerical_x_train)
        # get the constant colum names.
        constant_columns = [column for column in numerical_x_train.columns
                            if column not in numerical_x_train.columns[vs_constant.get_support()]]
        # detect constant categorical variables.
        constant_cat_columns = [column for column in X_train.columns 
                                if (X_train[column].dtype == "O" and len(X_train[column].unique())  == 1 )]
        all_constant_columns = constant_cat_columns + constant_columns
        X_train.drop(labels=all_constant_columns, axis=1, inplace=True)
        X_test.drop(labels=all_constant_columns, axis=1, inplace=True)
        print(X_train.shape)
        # threshold value for quasi constant.
        ####### Quasi-Constant Features
        threshold = 0.98
        # create empty list
        quasi_constant_feature = []
        # loop over all the columns
        for feature in X_train.columns:
            # calculate the ratio.
            predominant = (X_train[feature].value_counts() / np.float(len(X_train))).sort_values(ascending=False).values[0]
            # append the column name if it is bigger than the threshold
            if predominant >= threshold:
                quasi_constant_feature.append(feature) 
        X_train.drop(labels=quasi_constant_feature, axis=1, inplace=True)
        X_test.drop(labels=quasi_constant_feature, axis=1, inplace=True)
        print(X_train.shape)
        #######Duplicated Features
        # transpose the feature matrice
        train_features_T = X_train.T
          ########  Correlation Filter Methods
        # select the duplicated features columns names
        duplicated_columns = train_features_T[train_features_T.duplicated()].index.values
        # drop those columns
        X_train.drop(labels=duplicated_columns, axis=1, inplace=True)
        X_test.drop(labels=duplicated_columns, axis=1, inplace=True)
        print(X_train.shape)
        correlated_features = set()
        correlation_matrix = X_train.corr()
        for i in range(len(correlation_matrix .columns)):
            for j in range(i):
                if abs(correlation_matrix.iloc[i, j]) > 0.8:
                    colname = correlation_matrix.columns[i]
                    correlated_features.add(colname)
        X_train.drop(labels=correlated_features, axis=1, inplace=True)
        X_test.drop(labels=correlated_features, axis=1, inplace=True)
        print(X_train.shape)
        return X_train,X_test
    except:
        print('sucsessfully completed QC')