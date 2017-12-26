from __future__ import division
from cv2.ximgproc import guidedFilter
import Image
import math
import os
import numpy as np
import cv2 as cv2
import tensorflow as tf

def weightsInitialize(Shape):
	
	#random weights using Xavier Initialization
	W = tf.get_variable("W", shape = Shape, initializer=tf.contrib.layers.xavier_initializer())
	return W
	
def biasInitialize(Shape):
	B = tf.Variable(tf.constant(0.01, shape = Shape))
	return B
	
def convolutionFunc(ip, filter_size, no_channels, no_filters, activation, max_pool, dropout = None,  stride=(1,1)):
	
	#setting up the filter without bias
	weights = weightsInitialize([filter_size, filter_size, no_channels, no_filters])
    	
    	#convolution without bias
    	layer = tf.nn.conv2d(ip, filter = weights, strides = [1, stride[0], stride[1], 1], padding = 'SAME')
    	
    	if activation!= None:
    		layer = activation(layer)
    	
    	if max_pool:
    		layer = tf.nn.max_pool(layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    		
    	if dropout!= None:
    		layer = tf.nn.dropout(layer)
    		
    	return layer

	
def residualLayer(layer):
	
	#filter size needs to be changed here! refer to paper
	
	conv1 = convolutionFunc(layer, 3, 16, 16, activation = None, max_pool = False)
	batch_norm1 = tf.layers.batch_normalization(conv1)
	relu1 = tf.nn.relu(batch_norm1)
	
	conv2 = convolutionFunc(relu1, 3, 16, 16, activation = None, max_pool = False)
	batch_norm2 = tf.layers.batch_normalization(conv2)
	relu2 = tf.nn.relu(batch_norm2)
	
	conv3 = convolutionFunc(relu2, 3, 16, 16, activation = None, max_pool = False)
	batch_norm3 = tf.layers.batch_normalization(conv3)
	relu3 = tf.nn.relu(batch_norm3)
	
	output = relu3 + relu1
	
	return output
	

    	
def train(x,x_detail, y, learning_rate = 0.1):
	
	(m,h,w,c) = x.shape
	n_y = y.shape[1]
	x = tf.placeholder(tf.float32, [None, h, w, c], name='x')
	y = tf.placeholder(tf.float32, [None, n_y], name = 'y')
	
	
	prev_layer = x_detail
	
	for i in range(1):
		prev_layer = residualLayer(prev_layer)
	
	last_conv = convolutionFunc(prev_layer, 3, 16, 16, activation = None, max_pool = False)
	last_batch_norm = tf.layers.natch_normalization(last_conv)
	
	output_image = last_batch_norm + x

	
	cost = tf.reduce_mean((last_batch_norm + x -y)**2)
	path = os.getcwd()
	
	optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
	
	init = tf.global_variables_initializer()
	
	with tf.Session() as sess:
	
		sess.run(init)
		
		for j in range(m):
			sess.run([cost, optimizer], feed_dict = {x: x_detail, y: y,})
		
		for i in range(m):
			cv2.imwrite(path+"/train"+str(i+1)+".jpg",output_image[i])	
			


def image_slice_filter(image_path,outdir,image_name):
    img = Image.open(image_path)
    width, height = img.size
    left = 0
    top = 0
    right = int(width/2)
    bottom = height

    #Crop and resize non rain part of image
    crop_1 = (left, top, right, bottom)
    image_1 = img.crop(crop_1)
    image_1 = image_1.resize((64,64),Image.ANTIALIAS) 
    image_1.save(os.path.join(outdir+"/non_rain/",image_name+"_non_rain"+".png"))

    #Crop and resize rain part of image
    crop_2 = (right, top, width, bottom)
    image_2 = img.crop(crop_2)
    image_2 = image_2.resize((64,64),Image.ANTIALIAS) 
    image_2.save(os.path.join(outdir+"/rain/",image_name+"_rain"+".png"))

    #Filter the rain image
    img = cv2.imread(outdir+"/rain/"+image_name+"_rain"+".png")    
    guided = guidedFilter(img,img,15,0.2*255*255) 
    detail = img - guided
    cv2.imwrite(outdir+"/rain/"+image_name+"_rain"+"_guided.png",guided)
    cv2.imwrite(outdir+"/rain/"+image_name+"_rain"+"_detail.png",detail)
				
if __name__ == '__main__' :
	
	for i in range(5):
		image_slice_filter(os.getcwd()+"/training/"+str(i+1)+".jpg",os.getcwd()+"/split/",str(i+1))

	images=[]
	images_detail = []
	y_images = []
	path = os.getcwd()+"/split/rain"
	path_y = os.getcwd()+ "/split/non_rain"
	for i in range(5):
		images.append(cv2.imread(path+"/"+str(i+1)+"_rain.png"))
		images_detail.append(cv2.imread(path+"/"+str(i+1)+"_rain_detail.png"))
		y_images.append(cv2.imread(path_y+"/"+str(i+1)+"_non_rain.png"))

	x_train = np.asarray(images,dtype = np.float32)
	x_detail = np.asarray(images_detail,dtype = np.float32)
	y_train = np.asarray(y_images,dtype = np.float32)
		
#	y_train = 
	
	#x_train = #reduce dimension
	#y_train = #reduce dimension
	
	train (x_train,x_detail,y_train)
	#test ka bhi bana do bros