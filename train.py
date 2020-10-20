import cv2
import pandas as pd
import numpy as np
import os
import argparse
import json
import shutil
import matplotlib.pyplot as plt

from scipy import stats
from sklearn import metrics
from sklearn import model_selection
from sklearn import preprocessing
import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras import backend as K



def load_model_config(config_path):
    """
    Load the model json file from disc
    """
    full_path = os.path.expanduser(config_path)
    with open(full_path, 'r') as f:
        config = json.load(f)      
    
    return config


def make_model_folder(config, config_path):
	'''
	Make folder within dataset folder that will contain the model
	'''

	# Specifying path to save dataset
	dataset_path = os.path.expanduser(f"datasets/{config['data_process']['dataset_name']}/")
	model_path = os.path.expanduser(f"{dataset_path}models/{config['model']['model_name']}/")
	if os.path.exists(model_path):
		shutil.rmtree(model_path)

	os.makedirs(model_path)

	# Save a copy of config to dataset folder for reference
	shutil.copy(config_path, os.path.join(model_path, 'model_config.json'))

	return dataset_path, model_path


def load_data(dataset_path, model_path):
	'''	
	use grids.csv to get A_IDs and Y values. Use A_IDs to load road network images into pixel values to make X
	'''

	#read grid.csv into dataframe
	df_grid = pd.read_csv(dataset_path + 'grids.csv', index_col=0)

	A_ID_list = df_grid.columns.values  #list of A_IDs

	#split A_IDs into train and test
	A_ID_train, A_ID_test = model_selection.train_test_split(A_ID_list, test_size=0.2, random_state=100)

	#write train and test A_IDs to model folder for prediction comparison
	pd.DataFrame(A_ID_train).to_csv(model_path + 'A_ID_train.csv', index=False, header=False)
	pd.DataFrame(A_ID_test).to_csv(model_path + 'A_ID_test.csv', index=False, header=False)

	x_train = []
	y_train = []

	#iteratively construct x_train and y_train using A_ID, df_grd, and image folder
	for A_ID in A_ID_train:
		x_train.append(cv2.imread(dataset_path + 'images/' + A_ID + '_input.png', 0))
		y_train.append(df_grid[A_ID].values)

	x_test = []
	y_test = []

	#iteratively construct x_test and y_test using A_ID, df_grd, and image folder
	for A_ID in A_ID_test:
		x_test.append(cv2.imread(dataset_path + 'images/' + A_ID + '_input.png', 0))
		y_test.append(df_grid[A_ID].values)

	X_train = np.array(x_train)
	Y_train = np.array(y_train)
	X_test = np.array(x_test)
	Y_test = np.array(y_test)
	
	return X_train, Y_train, X_test, Y_test, A_ID_train, A_ID_test

def build_model(config, image_pixel_size, grid_length):
	'''
	Build keras model
	'''

	model = keras.Sequential()
	#add model layers

	for layer in config:
		arg = config[layer]  #arguments for this layer

		if arg['type'] == 'Conv2D_first':
			model.add(Conv2D(
				filters=arg['filters'],
				input_shape=(image_pixel_size, image_pixel_size, 1),
				kernel_size=(arg['kernel_size'], arg['kernel_size']),
				strides=(arg['strides'], arg['strides']),
				padding='same',
				activation=arg['activation']
				)
			)

		if arg['type'] == 'Conv2D':
			model.add(Conv2D(
				filters=arg['filters'],
				kernel_size=(arg['kernel_size'], arg['kernel_size']),
				strides=(arg['strides'], arg['strides']),
				padding='same',
				activation=arg['activation']
				)
			)

		if arg['type'] == 'MaxPooling2D':
			model.add(MaxPooling2D(
				pool_size=(arg['pool_size'], arg['pool_size']),
				strides=(arg['strides'], arg['strides']),
				)
			)

		if arg['type'] == 'Flatten':
			model.add(Flatten())

		if arg['type'] == 'FullyConnected':
			model.add(Dense(
				units=arg['units'],
				activation=arg['activation']
				)
			)

	print(model.summary())

	return model

def fit_model(X_train, Y_train, X_test, Y_test, config, image_pixel_size, grid_length, model_path):
	'''
	Fit CNN
	'''

	#build model framework
	model = build_model(config['layers'], image_pixel_size, grid_length)

	#add loss and optimizer to model
	model.compile(loss="binary_crossentropy", optimizer=keras.optimizers.Adam(lr=config['learning_rate']))

	#add model callbacks
	callbacks = [
		tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5),
		keras.callbacks.TensorBoard(log_dir=model_path)
	]

	model.fit(
		x = X_train,
		y = Y_train,
		batch_size = config['batch_size'],
		epochs = config['epochs'],
		verbose = 1,
		callbacks = callbacks,
		validation_data = (X_test, Y_test)
		)

	return model

def make_grid(grid_size, image_pixels):
	'''
	make dictionary of mini boxes representing each cell of grid
	'''
	
	#size of each grid sub box
	delta = image_pixels/grid_size

	mini_bboxes = []

	for i in range(0, grid_size):  #rows  
		for j in range(0, grid_size):   #columns
			mini_bbox = {}
			mini_bbox['north'] = delta*i
			mini_bbox['south'] = mini_bbox['north'] + delta
			mini_bbox['west'] = delta*j
			mini_bbox['east'] = mini_bbox['west'] + delta

			mini_bboxes.append(mini_bbox)

	return mini_bboxes
		
def add_grid(A_IDs, Y_pred, grid_size, image_pixels, dataset_path, model_path, folder):
	'''
	iterate through A_IDs and append image pixel array and grid array to X and Y
	'''

	os.mkdir(f'{model_path}{folder}_predict/')  #make folder to store predicted images
	grid_dict = {}  #make dictionary for converting to csv to compare danger coefficients

	A_ID_counter = 0
	for A_ID in A_IDs:
		grid_dict[A_ID] = Y_pred[A_ID_counter]     #add predicted danger coeffs to dict
		
		fig = plt.figure()
		ax = fig.add_subplot()

		#read in A-ID_input image to ax
		image = plt.imread(dataset_path + 'images/' + A_ID + '_input.png')
		ax.imshow(image)
		ax.axis('off')
		
		mini_bboxes = make_grid(grid_size, image_pixels)  #make dictionary of mini boxes representing each cell of grid

		#iterate through mini boxes and add danger coeff to each cell of grid
		box_counter = 0
		for mini_bbox in mini_bboxes:

			danger_coeff_pred = Y_pred[A_ID_counter][box_counter]

			#specify xs and ys for mini_bbox for plotting
			xs = [mini_bbox['west'], mini_bbox['west'], mini_bbox['east'], mini_bbox['east']]
			ys = [mini_bbox['north'], mini_bbox['south'], mini_bbox['south'], mini_bbox['north']]
			
			#messing with transparency to make plots easier to visualize
			alpha = danger_coeff_pred*5
			if alpha > 0.5:
				alpha = 0.5
			
			#generate plot overlay and add to ax
			ax.fill(xs, ys, "r", alpha = alpha)
			ax.text(xs[0], np.mean(ys), "%.8f" % danger_coeff_pred, color='green', fontsize=6)

			box_counter += 1

		A_ID_counter += 1
			
		fig.savefig(f'{model_path}{folder}_predict/{A_ID}_predict.png')
		plt.close() #clear figures

	pd.DataFrame(grid_dict).to_csv(f'{model_path}grid_{folder}_pred.csv')



def prediction_images(X_train, X_test, A_ID_train, A_ID_test, config, model, dataset_path, model_path):
	'''
	add prediction overlay onto initial image for comparison purposes
	'''

	Y_train_pred = model.predict(x = X_train, batch_size=config['model']['batch_size'])
	Y_test_pred = model.predict(x = X_test, batch_size=config['model']['batch_size'])

	add_grid(A_ID_train, Y_train_pred, config['data_process']['grid_size'], config['data_process']['image_pixels'], dataset_path, model_path, 'train')
	add_grid(A_ID_test, Y_test_pred, config['data_process']['grid_size'], config['data_process']['image_pixels'], dataset_path, model_path, 'test')


def main(args):

	#load 'data_process' parameters from model_config.json into dictionary
	config = load_model_config(args.config_path)

	#make folder for dataset, put copy of model_config.json there, and return data_path for saving images
	dataset_path, model_path = make_model_folder(config, args.config_path)
	print('Config loaded, model folder created:', model_path)

	#load data from folder of images
	X_train, Y_train, X_test, Y_test, A_ID_train, A_ID_test = load_data(dataset_path, model_path)
	print('Data loaded')

	#dataset properties
	image_pixel_size = X_train[0].shape[0]
	grid_length = Y_train[0].shape[0]
	num_train_examples = len(X_train)
	num_test_examples = len(X_test)

	#normalize X data
	X_train = X_train/255
	X_test = X_test/255

	#reshape for inputting in model
	X_train = X_train.reshape(num_train_examples, image_pixel_size, image_pixel_size, 1)
	X_test = X_test.reshape(num_test_examples, image_pixel_size, image_pixel_size, 1)

	#print properties of dataset
	print('X_shape', X_train.shape)
	print('Y_shape:', Y_train.shape)
	print('number train examples:', num_train_examples)
	print('number test examples:', num_test_examples)


	#Building and Training model
	print('Building and Training Model')
	model = fit_model(X_train, Y_train, X_test, Y_test, config['model'], image_pixel_size, grid_length, model_path)

	
	print('Model trained, making predictions')
	prediction_images(X_train, X_test, A_ID_train, A_ID_test, config, model, dataset_path, model_path)
	
	

	#for loss, use binary cross entropy
	#https://github.com/suraj-deshmukh/Keras-Multi-Label-Image-Classification/blob/master/model.py




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-cf',
        '--config_path',
        default='model_config.json',
        help="Path to model config file"
    )
    args = parser.parse_args()
    main(args)