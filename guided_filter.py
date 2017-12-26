import numpy as np
import cv2 as cv2
from cv2.ximgproc import guidedFilter
import os

if __name__ == '__main__':
    img = cv2.imread("test_2.png")    
    guided = guidedFilter(img,img,15,0.2*255*255) 
    detail = img - guided
    path = os.getcwd()
    cv2.imwrite(path+"/test_guided.jpg",guided)
    cv2.imwrite(path+"/test_detail.jpg",detail)
