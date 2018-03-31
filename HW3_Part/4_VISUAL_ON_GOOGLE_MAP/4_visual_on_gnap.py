
# coding: utf-8

# In[1]:

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


# In[ ]:



