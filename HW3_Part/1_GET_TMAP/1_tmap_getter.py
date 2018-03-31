
# coding: utf-8

# In[2]:

import urllib.request as ur

TMAP_APP_KEY = "0a556b9e-906f-335c-9e4b-c3ad9091a103"
MIN_LAT = 37.56
MIN_LON = 126.88
MAX_LAT = 37.57
MAX_LON = 126.89
ZOOM_LEVEL = 19

INT = 0.01

OUT_DIR = "./RESULT/"
NUM = 4


# TODO: use for loops to get the traffic info for 4 * 4 areas
# TODO: fill in the request string


for i in range(NUM):
    for j in range(NUM):
        AREA_INDEX  = NUM * i +j
        requestStr = "https://apis.skplanetx.com/tmap/traffic?version=1&minLat="+             str(MIN_LAT+INT*i) + "&minLon=" + str(MIN_LON+INT * j)+ "&maxLat="+ str(MAX_LAT+INT *i)+             "&maxLon="+ str(MAX_LON+INT * j)+"&zoomLevel="+ str(ZOOM_LEVEL)+"&appKey="+ TMAP_APP_KEY +            "&reqCoordType=WGS84GEO&resCoordType=WGS84GEO&format=xml"
        in_file = ur.urlopen(requestStr)
        out_file = open(OUT_DIR+'traffic_%02d.xml' % (AREA_INDEX), 'wb')
        out_file.write(in_file.read())
        out_file.close()


# In[ ]:



