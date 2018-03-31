
# coding: utf-8

# In[ ]:


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


from bs4 import BeautifulSoup
from urllib.request import urlopen

IN_DIR = "../1_GET_TMAP/RESULT/"
OUT_DIR = "./RESULT/"
for area_index in range(16):
    in_file = IN_DIR + 'traffic_%02d.xml' % (area_index)
    out_file = open(OUT_DIR+'congestion_coor_%02d.txt' % (area_index), 'w')
    f = open(in_file,encoding='UTF-8') 
    soup = BeautifulSoup(f,'html5lib')
    
    congestionlevel = []
    coordi = []
    
    for node in soup.findAll('tmap:congestion',attrs={'class': None}):  
        #print("congestion level : "+node.string)
        congestionlevel += node.string
        #print(congestionlevel)
        
    for node2 in soup.findAll('coordinates'):
        coordi += node2
        
    for x in range(len(soup.findAll('coordinates'))-1):  
        out_file.write(congestionlevel[x]+";"+coordi[x]+ "\n")

    out_file.close()


# In[ ]:


import webbrowser
#AIzaSyAzgANCMQVVwbJW42KEb_NyhGtTHoSpFXs
GOOGLE_APP_KEY = 'AIzaSyAzgANCMQVVwbJW42KEb_NyhGtTHoSpFXs'
CENTER = '37.60,126.90'

IN_DIR = '../2_PARSE_TMAP/RESULT/'
OUT_DIR = './RESULT/'

for area_index in range(16):
    in_file = open(IN_DIR + "congestion_coor_%02d.txt" % (area_index), 'r')
    out_file = open(OUT_DIR + "gmap_urlstr_%02d.txt" % (area_index), 'w')
    
    color = []
    fst = ''
    last = ''    
    mylist=[]
    mylist2 = []
    path =''
    cnt = 0
    
    for line in in_file:
        congestion, linestring = line.split(';')
        
        if congestion == '1':
            color.append('blue')
        elif congestion == '2':
            color.append('green')
        elif congestion == '3':
            color.append('yellow')
        elif congestion == '4':
            color.append('orange')
        elif congestion == '5':
            color.append('red')
            
        mylist.append(linestring[:-1].split(" "))
        
    for x in range(0,len(mylist)):
        for y in range(len(mylist[x])):
            dd, ee = mylist[x][y].split(",")
            mylist[x][y] = (str(ee) + ',' + str(dd))
    
    for p in range(len(color)):
        path += "&path=color:" +color[p] +"|weight:6|" +mylist[p][0] +'|' + mylist[p][len(mylist[p])-1]  
        
    urlstr = "https://maps.googleapis.com/maps/api/staticmap?center=" + CENTER+        "&zoom=12&size=800x400&maptype=roadmap&sensor=false&key=" + GOOGLE_APP_KEY + path   
        

    out_file.write(urlstr)
    
    out_file.close()

#last = linestring.split()[0] + linestring.split()[len(linestring.split())-1]


# In[ ]:


import webbrowser
#AIzaSyAzgANCMQVVwbJW42KEb_NyhGtTHoSpFXs
GOOGLE_APP_KEY = 'AIzaSyAzgANCMQVVwbJW42KEb_NyhGtTHoSpFXs'
CENTER = '37.60,126.90'

IN_DIR = '../3_SHOW_ON_GOOGLE_MAP/RESULT/'
OUT_DIR = './RESULT/'

for area_index in range(16):
    in_file = open(IN_DIR + "congestion_coor_%02d.txt" % (area_index), 'r')
    out_file = open(OUT_DIR + "gmap_urlstr_%02d.txt" % (area_index), 'w')
    
    color = [] 
    mylist=[]
    mylist2 = []
    path =''
    markers = ''
    cnt = 65
    
    for line in in_file:
        congestion, linestring = line.split(';')
        
        if congestion == '1':
            color.append('blue')
        elif congestion == '2':
            color.append('green')
        elif congestion == '3':
            color.append('yellow')
        elif congestion == '4':
            color.append('orange')
        elif congestion == '5':
            color.append('red')
            
        mylist.append(linestring[:-1].split(" "))
        
    for x in range(0,len(mylist)):
        for y in range(len(mylist[x])):
            dd, ee = mylist[x][y].split(",")
            mylist[x][y] = (str(ee) + ',' + str(dd))
    
    for p in range(len(color)):
        path += "&path=color:" +color[p] +"|weight:6|"+ 'fillcolor:'+ color[p] + '|'+mylist[p][0] +'|' + mylist[p][len(mylist[p])-1]  
        if p<10:
            markers+= '&markers=size:mid%7Ccolor:' + color[p]+ '%7Clabel:' + str(p) + '%7C' + mylist[p][0]
        else:
            markers+= '&markers=size:mid%7Ccolor:' + color[p]+ '%7Clabel:' + chr(cnt) + '%7C' + mylist[p][0]
            cnt += 1 
            
    urlstr = "https://maps.googleapis.com/maps/api/staticmap?center=" + mylist[p][0] +         "&zoom=14&size=800x400&maptype=hybrid&scale=2&sensor=false&key=" + GOOGLE_APP_KEY + path + markers  
        
    #webbrowser.open(urlstr)

    out_file.write(urlstr)
    
    out_file.close()

#last = linestring.split()[0] + linestring.split()[len(linestring.split())-1]

