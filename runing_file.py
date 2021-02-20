from datacleaning import DataCleaning 
from prediction import DataPrediction
from VcfFile import DC_PD
import argparse
from configparser import ConfigParser
from smart_open import smart_open 
from os.path import splitext
import pandas as pd
import joblib
parser = argparse.ArgumentParser()
parser.add_argument("--target", help="enter target feature", type = str)
parser.add_argument("--dataset", help="enter dataset name", type = str)
parser.add_argument("--prediction", help="enter dataset name", type = str)
parser.add_argument("--model_build", help="enter dataset name", type = str)
parser.add_argument("--model_file", help="enter dataset name", type = str)
parser.add_argument("--phenome_data", help="enter dataset name", type = str)
args = parser.parse_args()
i=args.target
dname = args.dataset
model_file = args.model_file
phenome = args.phenome_data
config_object = ConfigParser()
ini = config_object.read(r'config.ini')
config_object.read(ini)
userinfo = config_object["MYSQL"]
global access_key_id
global secret_access_key
access_key_id = userinfo["access_key_id"]
secret_access_key  = userinfo["secret_access_key"]
bucket  = userinfo["bucket"]
bucket_name = bucket
file_name,extension = splitext(dname)
path = 's3://{}:{}@{}/{}'.format(access_key_id, secret_access_key, bucket_name, dname)
pre_D = 's3://{}:{}@{}/{}'.format(access_key_id, secret_access_key, bucket_name, model_file)
pheN = 's3://{}:{}@{}/{}'.format(access_key_id, secret_access_key, bucket_name, phenome)
def csv_predicts(path,pre_D):
    print('file formate csv')
    df = pd.read_csv(smart_open(path))
    df1 = df.fillna(df.mean())
    df.isnull().sum()
    df1 = df1.sort_values(i)
    model_instance = DataCleaning(df1,i,ini = ini)
    model_instance.dtypes_handliing()
    print(model_instance.handiling_categorical())
    model_instance.handiling_int_col()
    cleaned_Data_frm= model_instance.concat_cat()
    cleaned_Data_frm1= model_instance.concat_int()
    y = model_instance.encoder()
    data = model_instance.classification(cleaned_Data_frm, cleaned_Data_frm1,y)
    model_instance = DataPrediction(data,i,pre_D)
    resutls = model_instance.split()
    return resutls
if extension == '.csv':
    results =  csv_predicts(path,pre_D)
elif extension == '.vcf':
    print('vcf')
    model_instance = DC_PD(path,pheN,i)
    model_instance.total_QC()