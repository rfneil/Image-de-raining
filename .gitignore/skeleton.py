import numpy as np

class Derain(object):

	path = ""

    def __init__(self, data_dir,checkpoint_dir='./checkpoints/'):
        """
            data_directory : path like /home/kushagr/NNFL_Project/rain/training/
            	includes the dataset folder with '/'
            Initialize all your variables here
        """

    def train(self, training_steps=10):
        """
            Trains the model on data given in path/train.csv
            	which conatins the RGB values of each pixel of the image  

            No return expected
        """
        

    def save_model(self, step):

        # file_name = params['name']
        # pickle.dump(self, gzip.open(file_name, 'wb'))

        """
            saves model on the disk
            You can use pickle or Session.save in TensorFlow
            no return expected
        """


    def load_model(self, **params):
    	# file_name = params['name']
        # return pickle.load(gzip.open(file_name, 'rb'))

        """
            returns a pre-trained instance of Segment class
        """
