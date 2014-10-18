#!/bin/bash

parallel --colsep ' ' ssh {1} python /home/yuncong/Brain/notebooks/pipeline_v3.py /home/yuncong/DavidData/RS141/x5/{2}/RS141_x5_{2}.tif redNissl  /home/yuncong/DavidData/RS141/x5/0001/redNissl/labelings/RS141_x5_0001_redNissl_models.pkl :::: argfile
