from __future__ import division
import Image
import math
import os

def split_slice(image_path,outdir,image_name):
    img = Image.open(image_path)
    width, height = img.size
    left = 0
    top = 0
    right = int(width/2)
    bottom = height

    crop_1 = (left, top, right, bottom)
    image_1 = img.crop(crop_1)
    image_1.save(os.path.join(outdir,image_name+"_"+ str(1)+".png"))
    crop_2 = (right, top, width, bottom)
    image_2 = img.crop(crop_2)
    image_2.save(os.path.join(outdir,image_name+"_"+ str(2)+".png"))
    

if __name__ == '__main__':
    #slice_size is the max height of the slices in pixels
    split_slice("test.jpg", os.getcwd(),"test")