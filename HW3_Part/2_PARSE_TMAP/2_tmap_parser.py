
# coding: utf-8

# In[37]:

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



