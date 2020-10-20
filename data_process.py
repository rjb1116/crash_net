import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import osmnx as ox
import json
import os
import shutil
import cv2

def load_model_config(config_path):
    """
    Load the model json file from disc
    """
    full_path = os.path.expanduser(config_path)
    with open(full_path, 'r') as f:
        config = json.load(f)
    
    return config['data_process']

def make_data_folder(config, config_path):
	'''
	Make dataset folder and images folder
	'''

	# Specifying path to save dataset
	dataset_path = os.path.expanduser(f"datasets/{config['dataset_name']}/")
	image_path = os.path.expanduser(dataset_path + 'images/')
	if os.path.exists(dataset_path):
		shutil.rmtree(dataset_path)

	os.makedirs(image_path)

	# Save a copy of config to dataset folder for reference
	shutil.copy( config_path, os.path.join(dataset_path, 'data_process_config.json'))

	return dataset_path, image_path


def get_inputs(config):
	'''
	Load accident csvs into dataframes and process them: remove entries with no zip codes, and remove entries with <=100 accidents in that zip code
	'''

	df_zip = pd.read_csv(config['zip_csv_path'], index_col=0)   #Import csv wtih total accidents per zip code
	df_zip['Zip_simple'] = df_zip['Zip_simple'].astype(str)  #convert Zip_simple columns to string to avoid comparison errors later on
	zips_100plus = df_zip[df_zip['tot_accidents'] > 100]['Zip_simple'].values #generate list of Zip_simples that have >100 accidents

	df = pd.read_csv(config['accidents_csv_path'])  #Import csv with accident database into pandas df (~3.5M rows)
	df['Zip_simple'] = df['Zipcode'].str[0:5] #add Zip_simple column with just 5 letter zip codes 
	df.dropna(subset=['Zip_simple'], inplace=True)  #drop rows from column that are missing zip codes (~1000)
	df = df[df['Zip_simple'].isin(zips_100plus)]  #filter out zip codes with <100 accidents

	# limiting database to a specific location
	df = df[df[config['limit_category']] == config['limit_value']]

	# if no A_ID specified, generate images for each accident in the df, otherwise generate for A_ID specified
	if config['A_ID'] == 'None':
		#randomly sample A_IDs from df to create training examples
		A_IDs = df['ID'].sample(n = config['sample_size'], replace = False).values
		print('No A_IDs specified, will generate images for', config['sample_size'], 'accidents in', config['limit_value'])  #ADD SAMPLE SIZE AND LIMIT VALUE
	else:
		A_IDs = [config['A_ID']]
		print('A_ID specified, will generate images for', A_IDs, 'in', config['limit_value'])
	
	#distance from center of graph to end
	distance = float(config['distance'])

	#grid_size N x N for danger zones
	grid_size = config['grid_size']

	#number of pixels on one size of image
	pixels = config['image_pixels']

	return df, df_zip, A_IDs, distance, grid_size, pixels


def get_bbox(ax):
	'''
	Get lat, lng at edges from a graph
	'''
	bbox = {}
	bbox['south'], bbox['north'] = ax.get_ylim()
	bbox['west'], bbox['east'] = ax.get_xlim()

	return bbox

def	make_graph(lat, lng, distance, pixels):
	'''
	Make initial graph of accident area from lat, lng, return graph, plot, and bounding box of area plotted
	'''
	# get graph of accident area
	G = ox.graph_from_point((lat, lng), dist=distance, dist_type='bbox', network_type='drive', retain_all=True, simplify=False)

	#create fig
	my_dpi = 192
	fig = plt.figure(figsize=(pixels/my_dpi, pixels/my_dpi), dpi=my_dpi, constrained_layout=True)
	ax = fig.add_subplot()
	
	#add graph to plot
	ox.plot_graph(G, node_size=0, edge_color = 'black', ax=ax, show=False)

	bbox = get_bbox(ax)

	return fig, ax, G, bbox


def get_accidents(bbox, df):
	'''
	Get df with just the accidents that occurred within a bbox
	'''

	df_bbox = df[(df['Start_Lat'] > bbox['south']) & (df['Start_Lng'] > bbox['west']) & (df['Start_Lat'] < bbox['north']) & (df['Start_Lng'] < bbox['east'])]

	return df_bbox


def add_accidents(fig, ax, bbox, df):
	'''
	Create array of accidents to plot on graph of accidents that occurred in the area
	Inputs
		lat: latitude [float]
		lng: longitude [float]
		bbox: latitude and longitudes that bound the area around the accident
		df: dataframe of accidents
	Outputs
		xs: xs for scatter plot
		ys: ys for scatter plot
	'''
	df_bbox = get_accidents(bbox, df)

	#add accident points
	ax.scatter(df_bbox['Start_Lng'].values, df_bbox['Start_Lat'].values, s=40, marker='.', linewidths=0, c='red', alpha = 0.5)

	return fig, ax, df_bbox


def make_mini_bboxes(bbox, grid_size):
	'''
	Split bounding box into mini bboxes based on grid_size
	'''

	#make mini bboxes defining grid
	del_lat = (bbox['north'] - bbox['south'])/grid_size
	del_lng = (bbox['east'] - bbox['west'])/grid_size
	
	mini_bboxes = []

	for i in range(0, grid_size):  #rows  
		for j in range(0, grid_size):   #columns
			mini_bbox = {}
			mini_bbox['north'] = bbox['north'] - del_lat*i
			mini_bbox['south'] = mini_bbox['north'] - del_lat
			mini_bbox['west'] = bbox['west'] + del_lng*j
			mini_bbox['east'] = mini_bbox['west'] + del_lng

			mini_bboxes.append(mini_bbox)		

	return mini_bboxes


def get_danger_coeff(df_mini_bbox, num_zip_accidents):
	'''
	Gets danger coefficient of a bbox by dividing number of accidents by total number of accidents in that zip code
	'''

	danger_coeff = len(df_mini_bbox)/num_zip_accidents

	return danger_coeff


def add_grid(fig, ax, grid_size, bbox, df_bbox, num_zip_accidents):
	'''
	Create grid of danger scores to overlay on graph of accident area. Normalize by total accidents reported in that zip code
	Outputs
		grid: numpy array of the danger score of each box of grid
	'''
	
	# Generate mini bounding box grid
	mini_bboxes = make_mini_bboxes(bbox, grid_size)

	# get danger value for each bbox of grid
	grid = []

	#calculate danger_coeff for each mini bbox and add weighted transpared box to plot
	for mini_bbox in mini_bboxes:
		#calculate danger_coeff
		df_mini_bbox = get_accidents(mini_bbox, df_bbox)
		danger_coeff = get_danger_coeff(df_mini_bbox, num_zip_accidents)
		grid.append(danger_coeff)

		#specify xs and ys for mini_bbox for plotting
		xs = [mini_bbox['west'], mini_bbox['west'], mini_bbox['east'], mini_bbox['east']]
		ys = [mini_bbox['north'], mini_bbox['south'], mini_bbox['south'], mini_bbox['north']]
		#messing with transparency to make plots easier to visualize
		alpha = danger_coeff*5
		if alpha > 0.5:
			alpha = 0.5
		#generate plot overlay and add to ax
		ax.fill(xs, ys, "r", alpha = alpha)
		ax.text(xs[0], np.mean(ys), "%.8f" % danger_coeff, color='blue', fontsize=3)


	return grid, ax	



def generate_single_accident(df, df_zip, A_ID, distance, grid_size, pixels, image_path):
	'''
	This script takes an accident ID from the Accidents csv and generates an input figure and a labelled figure
	'''

	#lat and lng of accident
	lat = df[df['ID'] == A_ID]['Start_Lat'].values[0]
	lng = df[df['ID'] == A_ID]['Start_Lng'].values[0]
	#lat = 37.807072
	#lng = -122.405352
	
	#zip code of accident
	zip_code = df[df['ID'] == A_ID]['Zip_simple'].values[0]

	#number of accidents reported in that zip code
	num_zip_accidents = df_zip[df_zip['Zip_simple'] == zip_code]['tot_accidents'].values[0]

	# make initial graph accident window and plot
	fig, ax, G, bbox = make_graph(lat, lng, distance, pixels)

	# save this as input figure
	fig.savefig(image_path + A_ID + '_input.png')

	#get df of accidents in accident window and add to figure
	fig, ax, df_bbox = add_accidents(fig, ax, bbox, df)

	#get grid with danger coefficient and overlay on figure
	grid, ax = add_grid(fig, ax, grid_size, bbox, df_bbox, num_zip_accidents)

	# save as labelled figure
	fig.savefig(image_path + A_ID + '_output.png')

	plt.close() #clear figures

	return grid


def main(args):

	#load 'data_process' parameters from model_config.json into dictionary
	config = load_model_config(args.config_path)

	#make folder for dataset, put copy of model_config.json there, and return data_path for saving images
	dataset_path, image_path = make_data_folder(config, args.config_path)
	print('Config loaded, dataset folder created:', dataset_path) 	

	# get inputs from provided arguments
	df, df_zip, A_IDs, distance, grid_size, pixels = get_inputs(config)
	print('Inputs Loaded Successfully')
	
	# initialize grid dict to more easily 
	grid_dict = {}
	# initialize array to capture A_IDs that fail processing
	failures_IDs = []
	
	# initialize counters for progress tracking
	num_rows = len(A_IDs)
	print('Num row to iterate:', num_rows)
	row_counter = 0
	threshold = 0
	failure_counter = 0
	

	#iterate through accidents
	for A_ID in A_IDs: 
		
		try:
			grid = generate_single_accident(df, df_zip, A_ID, distance, grid_size, pixels, image_path)
			grid_dict[A_ID] = grid	
		except:
			# osmnx sometimes can't generate the road network of an accident for various reasons, so count the number of times this happens and store the incident A-ID
			failure_counter += 1
			failures_IDs.append(A_ID)
		
		# track progress
		row_counter += 1
		percent_done = row_counter/num_rows*100
		if percent_done > threshold:
			print('Row:', row_counter, ', Perc done:', "%.0f" % percent_done, ', Num failures:', failure_counter, ', Perc failed', "%.2f" % (failure_counter/row_counter*100))
			threshold += 1

	#export danger matrix to grids.csv		
	pd.DataFrame(grid_dict).to_csv(dataset_path + 'grids.csv')

	#export list of failed A_IDs for manual examination
	pd.DataFrame(failures_IDs).to_csv(dataset_path + 'failed_IDs.csv')

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

