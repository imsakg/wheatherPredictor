#!/usr/bin/env python3

import pandas as pd
from pathlib import Path
import os

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

TEST= True
TESTMAX = 300
csvS = sorted(list(Path("datas").glob("*.csv")), key=os.path.getmtime)

'''
    # Date Time             :   01.01.2009 00:10:00
    # p (mbar)              :   996.52
    # T (degC)              :   -8.02
    # Tpot (K)              :   265.4
    # Tdew (degC)           :   -8.9
    # rh (%)                :   93.3
    # VPmax (mbar)          :   3.33
    # VPact (mbar)          :   3.11
    # VPdef (mbar)          :   0.22
    # sh (g/kg)             :   1.94
    # H2OC (mmol/mol)       :   3.12
    # rho (g/m**3)          :   1307.75
    # wv (m/s)              :   1.03
    # max. wv (m/s)         :   1.75
    # wd (deg)              :   152.3
'''

directions = {
	"N" : 45/4*0,
	"NBE" : 45/4*1,
	"NNE" : 45/4*2,
	"NEBN" : 45/4*3,
	"NE" : 45/4*4,
	"NEBE" : 45/4*5,
	"ENE" : 45/4*6,
	"EBN" : 45/4*7,
	"E" : 45/4*8,
	"EBS" : 45/4*9,
	"ESE" : 45/4*10,
	"SEBE" : 45/4*11,
	"SE" : 45/4*12,
	"SEBS" : 45/4*13,
	"SSE" : 45/4*14,
	"SBE" : 45/4*15,
	"S" : 45/4*16,
	"SBW" : 45/4*17,
	"SSW" : 45/4*18,
	"SWBS" : 45/4*19,
	"SW" : 45/4*20,
	"SWBW" : 45/4*21,
	"WSW" : 45/4*22,
	"WBS" : 45/4*23,
	"W" : 45/4*24,
	"WBN" : 45/4*25,
	"WNW" : 45/4*26,
	"NWBW" : 45/4*27,
	"NW" : 45/4*28,
	"NWBN" : 45/4*29,
	"NNW" : 45/4*30,
	"NBW" : 45/4*31
}

li = []

def fah2c(x):
    x= int(x[:2])
    x= (x-32)*5/9
    return str(f"{x:.2f}")

def mph2kmh(x):
    x= int(x[:2])
    return x*1.609344

def inhpa(x):
    x = float(x[:5])
    x *= 33.86
    return x #? str(f"{x:.2f}")

def wind(x):
    try:
        x = int(x[:2])
    except:
        x = int(x[:1])
    return x*2.237
latestC = ""

def compass(x):
    global latestC
    if x=="CALM" or x=="VAR":
        x = latestC
    else:
        latestC = x
    return directions[x]

def addDate(x,*args,**kwargs):
    if x!="nan":
        if "PM" in str(x):
            x = str(x).replace("PM","")
            xa = str(x).split(":")
            xf = int(xa[0]) + 12
            xf = "12" if xf==24 else xf
            x = str(xf)+":"+str(xa[1])
        
        elif "AM" in str(x):
            x = str(x).replace("AM","")
            xa = str(x).split(":")
            xf = int(xa[0])
            if xf<10:
                xf = "0"+str(xf)
            if xf == 12:
                xf= "00"
            x = str(xf)+":"+str(xa[1])

        #print(x + str(args[0]))
        x = str(args[0]) + " " + x
        return x[:-1]+":00"

if TEST:
    for i,filename in enumerate(csvS):
        df = pd.read_csv(filename, index_col=None, header=0).dropna()
        df = pd.DataFrame(df)
        filename = str(filename).replace(".csv","").replace("-",".").replace("datasEx/","").split(".")
        filename = filename[2]+"."+filename[1]+"."+filename[0]
        try:
            df["Time"] = df["Time"].apply(addDate,args=(filename,))
            li.append(df)
        except:
            pass
        if i>TESTMAX:
            break
else:
    for filename in csvS:
        df = pd.read_csv(filename, index_col=None, header=0).dropna()
        df = pd.DataFrame(df)
        filename = str(filename).replace(".csv","").replace("-",".").replace("datasEx/","").split(".")
        filename = filename[2]+"."+filename[1]+"."+filename[0]
        try:
            df["Time"] = df["Time"].apply(addDate,args=(filename,))
            li.append(df)
        except:
            pass
frame = pd.concat(li, axis=0, ignore_index=True)
#frame = frame.iloc[:, :-5]
#frame = frame.iloc[: , 1:]
frame = frame.drop(frame.columns[0], axis=1)
frame = frame.drop(frame.columns[-2], axis=1)
frame = frame.drop(frame.columns[-3], axis=1)
frame = frame.drop(frame.columns[-1], axis=1)
frame = frame.dropna()

frame["Temperature"] = frame["Temperature"].apply(fah2c)
frame["Dew Point"] = frame["Dew Point"].apply(fah2c)
frame["Humidity"] = frame["Humidity"].apply(lambda x: x[:2])
frame["Pressure"] = frame["Pressure"].apply(inhpa)
frame["Wind Speed"] = frame["Wind Speed"].apply(wind)
frame["Wind"] = frame["Wind"].apply(compass)

frame.to_csv("data.csv",index=False)
print("Success")