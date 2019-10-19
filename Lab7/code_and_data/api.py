
def geoGrab():

    import json  
    import urllib.request 

    j = 0
    f = open(r'Restaurant_Data_Beijing.txt','w') 

    for j in range (0,20):

        a = 'http://api.map.baidu.com/place/v2/search?q=%E9%A5%AD%E5%BA%97&page_size=20&page_num='
        b = '&region=%E5%8C%97%E4%BA%AC&output=json&ak=6fqat6X5NQl7cw7CAz0eHyATnRWInMSr'

        #����ĺ���(�ٷֺŲ���)����urlencode������ԭ���ǡ����ꡱ�͡�������
        #��Կ��Ҫ�Լ����룬Ȼ���滻������ġ���Կ��
        c = str(j)
        url = a+c+b
        j = j+1

        #url='http://api.map.baidu.com/place/v2/search?q=%E9%A5%AD%E5%BA%97&page_size=20'+
        #'&page_num=19&region=%E5%8C%97%E4%BA%AC&output=json&ak=qUPyb0ZPGmT41cL9L5irQzcnc48yIEck'
        temp = urllib.request.urlopen(url) 
             
        #���ַ���������ΪPython����
        hjson = json.loads(temp.read().decode('utf-8'))
        i = 0

        for i in range (0,20):

            lat = hjson['results'][i]['location']['lat']
            lng = hjson['results'][i]['location']['lng']
            print('%s\t%f\t' % (lat,lng))
            f.write('%s\t%f\t\n' % (lat,lng))
            i = i+1
    f.close()

geoGrab()
