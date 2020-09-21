import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import osmnx as ox


def get_inputs(args):

	df = pd.read_csv(args.csv_name).reset_index(drop=True)  #Import csv with accident database into pandas df
	df_zip = pd.read_csv('Accidents_by_zip.csv', index_col=0)   #Import csv wtih total accidents per zip code

	#distance from center of graph to end
	distance = float(args.distance)

	#grid_size N x N for danger zones
	grid_size = args.grid_size

	return df, df_zip, distance, grid_size


def get_bbox(ax):
	'''
	Get lat, lng at edges from a graph
	'''
	bbox = {}
	bbox['south'], bbox['north'] = ax.get_ylim()
	bbox['west'], bbox['east'] = ax.get_xlim()

	return bbox

def	make_graph(lat, lng, distance):
	'''
	Make initial graph of accident area from lat, lng, return graph, plot, and bounding box of area plotted
	'''
	# get graph of accident area
	G = ox.graph_from_point((lat, lng), dist=distance, dist_type='bbox', network_type='drive', retain_all=True, simplify=False)

	#create fig
	my_dpi = 192
	pixels = 400
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


	return grid, ax	



def generate_single_accident(df, df_zip, A_ID, distance, grid_size):
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
	fig, ax, G, bbox = make_graph(lat, lng, distance)

	# save this as input figure
	fig.savefig('figs/' + A_ID + '_input.png')

	#get df of accidents in accident window and add to figure
	fig, ax, df_bbox = add_accidents(fig, ax, bbox, df)

	#get grid with danger coefficient and overlay on figure
	grid, ax = add_grid(fig, ax, grid_size, bbox, df_bbox, num_zip_accidents)

	# save as labelled figure
	fig.savefig('figs/' + A_ID + '_output.png')

	plt.close() #clear figures

	return grid


def main(args):

	df, df_zip, distance, grid_size = get_inputs(args)
	print('Inputs Loaded Successfully')
 
	# looking at only San Francisco accidents
	df = df[df['City'] == 'San Francisco']
	
	# initialize grids array which will capture the output danger matrix
	grid_dict = {}
	
	num_rows = len(df)
	print('Num row to iterate:', num_rows)
	row_counter = 0
	threshold = 0
	failure_counter = 0
	failures_IDs = []

	#iterate through every accident
	for index, row in df.iterrows():
 
		A_ID = row['ID']
		
		try:
			grid = generate_single_accident(df, df_zip, A_ID, distance, grid_size)
			grid_dict[A_ID] = grid	
		except:
			# osmnx sometimes can't generate the image from different reasons, count the number of times this happens and store the incident A-ID
			failures += 1
			failures_IDs.append(A_ID)
	
		# track progress
		row_counter += 1
		percent_done = row_counter/num_rows*100
		if percent_done > threshold:
			print('Row:', row_counter, ', Perc done:', "%.0f" % percent_done, ', Num failures:', failure_counter, ', Perc failed', "%.2f" % (failure_counter/row_counter*100))
			threshold += 1
			

	pd.DataFrame(grid_dict).to_csv('grids.csv')
	pd.DataFrame(failures_IDs).to_csv('failed_IDs.csv')


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-c', '--csv_name', default='US_Accidents_zip_100plus.csv')
	parser.add_argument('-a', '--A_ID', default='A-809')
	parser.add_argument('-d', '--distance', default=250)
	parser.add_argument('-g', '--grid_size', default=6)
	args = parser.parse_args()
	main(args)
