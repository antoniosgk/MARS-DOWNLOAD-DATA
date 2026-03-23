#%%
'''
INFO ABOUT THE SCRIPT:
########################################################################

This script is used in order to download data
from ECMWF MARS with web api:

https://confluence.ecmwf.int/display/WEBAPI/Web+API+FAQ

This version has been developed for downloading
MARS data from analysis and from pressure levels 
which are representative for the entire atmospheric column

In order to find the experiments visit the following link:

https://apps.ecmwf.int/mars-catalogue/?class=rd 

Then go to (it is just an example):

https://apps.ecmwf.int/mars-catalogue/?type=an&class=rd&stream=oper&expver=hprj

Select atmospheric model:

https://apps.ecmwf.int/mars-catalogue/?type=an&class=rd&stream=oper&expver=hprj

Select either forecast or analysis links!!!

Here we focus on the analysis!!!

You can select among the surface and the pressure (or the model) levels!!!

Here we focus on the surface which are available at:

https://apps.ecmwf.int/mars-catalogue/?levtype=sfc&type=an&class=rd&stream=oper&expver=hprj

Select your preferences from the step, time and parameter fields!!!

At the links below you can specify your request and you will find 
all the information needed to run this script (see below)

Here you can monitor your submissions:

https://apps.ecmwf.int/webmars/joblist/ 

WEB api FAQ

https://confluence.ecmwf.int/display/WEBAPI/Web+API+FAQ


#Limitations:
20 queued requests per user
2 active requests per user

'''
from ecmwfapi import *
import pandas as pd
import datetime as dt
import os
import logging
import sys
from ecmwfapi import ECMWFService
logger=logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')

###############################################################################
scriptstart=dt.datetime.now()

print ('Script starts at:',scriptstart)
###############################################################################

#Here are the access credentials!!!
#server = ECMWFDataServer(url="https://api.ecmwf.int/v1",key="a04ef06e5c69ff77ebf5ce0294a5e011",email="zerefos@geol.uoa.gr") # feb tou 2024 kai march 2025 allaksame to klidi

server = ECMWFService("mars")


# Create the dates that you want to extract the ECMWF MARS data
# Starting date
fday   = "01"
fmonth = "06"
fyear  = "2025" 
strfdate = (fyear,fmonth,fday)

sday   = "02"
smonth = "06" 
syear  = "2025"
strsdate = (syear,smonth,sday)

freqdate = 'D' #Temporal frequency

fdate=dt.datetime.strptime("".join(strfdate),'%Y%m%d')
sdate=dt.datetime.strptime("".join(strsdate),'%Y%m%d')

#Create the timeseries for the period of interest
runperiod = pd.date_range(fdate,sdate,freq=freqdate)

rundates = [kk.strftime('%Y-%m-%d') for kk in runperiod]
classs  = "rd"
type    = "fc"
stream  = "oper"
expver  = "ilrf" #The acronym of the experiment
levtype = "ml"
levelist = "46/47/48/49/50/51/52/53/54/55/56/57/58/59/60"  # 46/47/48/49/50/51/52/53/54/55/56/57/58/59/60 #113/114/115/116/117/118/119/120/121/122/123/124/125/126/127/128/129/130/131/132/133/134/135/136/137
#levelist = "113/114/115/116/117/118/119/120/121/122/123/124/125/126/127/128/129/130/131/132/133/134/135/136/137"
#repres  = "sh"
grid    = "0.4/0.4" #Spatial resolution
area    = "90/-180/-90/180" #Domain N/W/S/E
param   = "210123/210203" #Parameters 228.128-percipitation - 210121/210122/210123/210203 #no2/_/co/o3
step    = "3/6/9/12/15/18/21/24" #3/6/9/12/15/18/21/24/27/30/33/36/39/42/45/48/51/54/57/60/63/66/69/72/75/78/81/84/87/90/93/96/99/102/105/108/111/114/117/120 
#step    = "3/6/9/12"
times   = ["00:00:00"] #Model initialization time
#times   = ["00:00:00","06:00:00","12:00:00","18:00:00"] #Model initialization time
#times   = ["06:00:00","18:00:00"]
format  = "netcdf"
#Create the folder where the data will be stored
datafolder='/home/agkiokas/MARS/data/'+expver+'/'+type.upper()+'/'+levtype.upper()


if not os.path.exists(datafolder):
    os.makedirs(datafolder)
    
##Download data for the selected dates
try:
    for rundate in rundates:
        
        for time in times:
                    
            outfolder=datafolder+'/'+rundate.replace('-','')+'/'
            
            if not os.path.exists(outfolder):
                os.makedirs(outfolder)
                
            outfile=expver+'_'+type+'_'+levtype+'_'+rundate.replace('-','')+time[:2]+".nc"           

            server.execute(
                {
                "class"      : classs,
                "type"       : type,
                "stream"     : stream,
                "expver"     : expver,
                "levtype"    : levtype,
                "levelist"   : levelist,
                "grid"       : grid,
                "area"       : area,
                "param"      : param,
                "step"       : step,
                "date"       : rundate,
                "time"       : time,
                "expect"     : "any",
                "format"     : format
                }, 
                os.path.join(outfolder,outfile))

except Exception as exc:
    print(f"ERROR for {rundate} {time}: {exc}")
    raise
    
    logstmp='logs_'+expver+'_'+type+'_'+levtype+'_'+dt.datetime.strftime(dt.datetime.now(),"%H%M%S%d%m%Y")
    
    
    logging.basicConfig(filename=logstmp+'.txt', level=logging.DEBUG, format='%(asctime)s %(levelname)s %(name)s %(message)s')
    logger=logging.getLogger(__name__)
    
    logging.exception('Oh no! Error. Here is the traceback info:')
        
    # #logging.critical(exc,exc_info=True)
    # print ("Error in:  ",expver+'_'+type+'_'+levtype+'_'+rundate+time[:2]+".nc")
        
    # stdout_handler = logging.StreamHandler(sys.stdout)
    # stdout_handler.setLevel(logging.DEBUG)
    # stdout_handler.setFormatter(formatter)

    # logstmp='logs_'+expver+'_'+type+'_'+levtype+'_'+dt.datetime.strftime(dt.datetime.now(),"%H%M%S%d%m%Y")

    # file_handler = logging.FileHandler(logstmp+'.log')
    # file_handler.setLevel(logging.DEBUG)
    # file_handler.setFormatter(formatter)
    
    # logger.addHandler(file_handler)
    # logger.addHandler(stdout_handler)
   
###############################################################################
scriptend=dt.datetime.now()

print ("")
print ("")
print ("")
print ("The script ended at:", scriptend)
print ("")
print ("")
print ("")
print ("The execution time was:", scriptend-scriptstart)
print ("")
print ("")
print ("")


#%%
##  ecmwf_mars_request_ml_fc.py
# %%
