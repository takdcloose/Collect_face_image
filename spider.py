import os
import cv2
from PIL import Image
import imagehash
from urllib.request import Request
from urllib.request import urlopen
from urllib.error import HTTPError
import time
from googlesearch import search
from bs4 import BeautifulSoup
import re
import tqdm
from face_recognize import face_detect,face_recog
from glob import glob

downloadDirectory = "./Download/img/"
name = "Scarlett Johansson image"
num = 1
threshold = 2

def getAbsoluteURL(baseUrl, source):
    if source.startswith('http://www.'):
        url = 'http://{}'.format(source[11:])
    elif source.startswith('https://www.'):
        url = 'http://{}'.format(source[12:])
    elif source.startswith('http://') or source.startswith('https://'):
        url = source
    elif source.startswith('www.'):
        url = source[4:]
        url = 'http://{}'.format(source)
    else:
        url = '{}/{}'.format(baseUrl, source)
    if not re.match(".*\.(jpg|png|bmp)",url):
        return None
    return url

def getExternalLinks(page):
    externalLinks = []
    for url in search(name, lang="jp", start=(page-1)*10, stop=10,pause = 2.0):
        externalLinks.append(url)
    return externalLinks

def delete_same_image():
    image_list = glob(downloadDirectory+"*.*")
    hash_dic = {}
    delete_list = []
    filename = image_list[0].split("\\")[-1]
    hash_dic[filename] = imagehash.phash(Image.open(image_list[0]))
    for i in tqdm.tqdm(range(1,len(image_list)), total=len(image_list)-1):
        filename = image_list[i].split("\\")[-1]
        for j in list(hash_dic.values()):
            diff = abs(imagehash.phash(Image.open(image_list[i]))-j)
            if diff > 2:
                hash_dic[filename] = imagehash.phash(Image.open(image_list[i]))
            else:
                delete_list.append(image_list[i])
    for i in delete_list:
        if os.path.exists(i):
            os.remove(i)
    return None

def DownloadImage(externalLinks):
    for url in externalLinks:
        global num
        header = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2228.0 Safari/537.3'}
        req = Request(url, headers=header)
        try:
            html = urlopen(req).read()
        except HTTPError as err:
            if err.code == 404:
                print("Page was not found")
                return None
            elif err.code == 403:
                time.sleep(2)
                print("bad behavior")
                return None
            else:
                raise
        bs = BeautifulSoup(html, 'html.parser')
        #get path of image
        downloadList = bs.find_all('img')
        for download in downloadList:
            try:
                #convert relative path to absolute path
                fileUrl = getAbsoluteURL(url, download['src'])
            except:
                continue
            if fileUrl is not None:
                #get all faces in the picture
                face_list = face_detect(fileUrl)
                print(fileUrl)
                if face_list is None:
                    continue
                for face in face_list:
                    #judge face
                    result = face_recog(face)
                    true_num = 0
                    #when the number of True is more than the threshold, write the image in local
                    for i in result:
                        true_num += i*1
                    if true_num >= threshold:
                        cv2.imwrite(downloadDirectory+str(num)+".jpg", face)
                        num += 1
            time.sleep(1)
    return None

#for page in range(1,15):
if __name__ == "__main__":

    externalLinks = getExternalLinks(1)
    DownloadImage(externalLinks)

    print("Complete to download")
    print("Delete the same image...")
    delete_same_image()
    print("Finish!")

