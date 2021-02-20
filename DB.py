from configparser import ConfigParser
import pymysql
import sqlalchemy
def localdb():
    config_object = ConfigParser()
    ini = config_object.read(r'config.ini')
    config_object.read(ini)
    config_object.read(ini)
    userinfo = config_object["MYSQL"]
    password= userinfo["password"]
    user= userinfo["user"]
    host= userinfo["host"]
    db= userinfo["db"]
    conn = pymysql.connect(host=(', '.join(["%s" % host])),
               port=int(3306),
               user=(', '.join(["%s" % user])),
               passwd=(', '.join(["%s" % password])),
               db=(', '.join(["%s" % db])), charset='utf8mb4')
    print("db connection success")
    cursor=conn.cursor()
    return cursor ,conn