import pandas as pd

fileCsv = pd.read_csv('spambase.csv')

fileMay = pd.read_csv('attack_ts_march_week3.csv', nrows = 10)
print(pd.DataFrame(fileMay))
#print (pd.DataFrame(fileCsv))

#a = 3
#b = 4
#print("test on pycharm")
#print(a+b)