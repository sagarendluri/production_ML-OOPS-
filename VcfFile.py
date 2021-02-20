from datacleaning import DataCleaning 
from prediction import DataPrediction
import allel
import pandas as pd
import numpy as np
from smart_open import smart_open 
class DC_PD():
    def __init__(self,path,pheN,i):
        self.path = path
        self.pheN = pheN
        self.i = i
    def total_QC(self):
        print("file type is VCF")
        callset = allel.read_vcf(smart_open(self.path))
        snps = callset['variants/ID']
        da = callset['calldata/GT']
        data = da.transpose([2,0,1]).reshape(-1,da.shape[1])
        df = pd.DataFrame(data)
        a = len(df)
        h = int(a/2)
        sm = callset['samples']
        def split(df) :
            hd = df.head(h)
            tl = df.tail(len(df)-h)
            return hd,tl
        # Split dataframe into top 3 rows (first) and the rest (second)
        heads,tails  = split(df)
        df1 = pd.DataFrame(heads)
        df2 = pd.DataFrame(tails)
        df1.columns = sm
        df2.columns = sm
        df1['snps'] = snps
        df2['snps'] = snps
        sum_df = df1.set_index('snps').add(df2.set_index('snps'), fill_value=0).reset_index()
        sum1_df = sum_df.set_index('snps')
        sum_df = sum1_df.T
        sum_df= sum_df.reset_index()
        # vcf file is ready #
#         sum_df.to_csv('vcfilfe')
#         path = 's3://{}:{}@{}/{}'.format(access_key_id, secret_access_key, bucket_name, object_key)
        phe  = pd.read_csv(smart_open(self.pheN))
        phe = phe.fillna(phe.mean())
        new_phe = phe.set_index('index')
        sum1 = sum_df.set_index('index')
        df3 = pd.merge(sum1, new_phe, left_index=True, right_index=True)
        final = df3.reset_index()
        final = final.sort_values('1st_Layer_Clusters')
        final1 = final.replace(to_replace =-2,value ='NaN') 
        final1 = final1.replace(to_replace =-1,value ='NaN')
        final1 = final1.replace({'NAN':np.nan}).astype(float)
        final1 = final.fillna(final.mean())
        df1 = final.sort_values(self.i)
        i = self.i
        model_instance = DataCleaning(df1,i,ini = ini)
        model_instance.dtypes_handliing()
        print(model_instance.handiling_categorical())
        model_instance.handiling_int_col()
        cleaned_Data_frm= model_instance.concat_cat()
        cleaned_Data_frm1= model_instance.concat_int()
        y = model_instance.encoder()
        data = model_instance.classification(cleaned_Data_frm, cleaned_Data_frm1,y)
        path = 's3://{}:{}@{}/{}'.format(access_key_id, secret_access_key, bucket_name, object_key)
        model_instance = DataPrediction(data,i,path)
        results = model_instance.split()
        return results