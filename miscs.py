# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import cv2
import os, sys

margin_border_effect = 200
for i in range(240,250):
    img = cv2.imread('/home/yuncong/my_csd181/ParthaData/PMD1305_reduce2/PMD1305_%d.reduce2.tif'%i, 0)
    region = img[2600-margin_border_effect:3380+margin_border_effect, 
                 2040-margin_border_effect:4300+margin_border_effect]
#     os.mkdir('/home/yuncong/my_csd181/ParthaData/PMD1305_reduce2/region1')
    cv2.imwrite('/home/yuncong/my_csd181/ParthaData/PMD1305_reduce2/region1/PMD1305_%d.reduce2.region1.tif'%i, region)
        
# tiffcrop -U px -z 1,1,2048,2048:1,2049,2048,4097 -e separate CheckScans.tiff Check

# <codecell>

for i in range(159,176):
    img = cv2.imread('/home/yuncong/my_csd181/ParthaData/PMD1305_reduce2/PMD1305_%d.reduce2.tif'%i, 0)
    region = img[1550-margin_border_effect:1550+2280+margin_border_effect, 
                 2320-margin_border_effect:2320+1570+margin_border_effect]
    cv2.imwrite('/home/yuncong/my_csd181/ParthaData/PMD1305_reduce2/region2/PMD1305_%d.reduce2.region2.tif'%i, region)

# <codecell>

for i in range(1,20):
    img = cv2.imread('/home/yuncong/my_csd181/ParthaData/PMD1305_reduce2/PMD1305_%d.reduce2.tif'%i, 0)
    if img is None: continue
    region = img[2100-margin_border_effect:2100+1200+margin_border_effect, 
                 2100-margin_border_effect::2100+820+margin_border_effect]
    cv2.imwrite('/home/yuncong/my_csd181/ParthaData/PMD1305_reduce2/region3/PMD1305_%d.reduce2.region3.tif'%i, region)

