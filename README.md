# crash_net
Predict traffic accidents from an image of the road network

## Quick Summary
This is the initial commit, mainly focused on data processing, of a crashNET neural network I'm building to predict traffic accident hotspots from images of road networks.

The python file data_process.py that's in the repo will generate the input image and output matrix I'm planning to use to train my CNN.

A long writeup of this proposed project can be found here: (https://www.reubenbritto.com/tdi-proposal)

## The data I'm processing

The data I'm processing comes from two sources: a database (https://smoosavi.org/datasets/us_accidents) of 3.5 million US accidents recorded from February 2016 to June 2020, and the osmnx python package (https://geoffboeing.com/2016/11/osmnx-python-street-networks/) that can generate node/edge graphs of the road network anywhere in the US.


## How to run the code

First, I recommend creating a virtual environment using condas specifically for this data processing since osmnx requires a lot of dependencies. Once you've installed condas, run the code below to create an (ox) virtual environment.

```
conda config --prepend channels conda-forge
conda create -n ox --strict-channel-priority osmnx
```

To run the data_process.py script, you'll need two things: 
* Accidents_by_zip.csv: this file is provided when you clone the repo. It's a tally of the number of accidents by zip code
* US_Accidents_June20.csv: this file you can download from [here](https://osmnx.readthedocs.io/) (this is the database from smoosavi referenced above)

If you run the following:
 
```
python data_process.py
```

the code will start generating input and output images for all the accidents that have occurred in SF and put them in the figs/ folder. This will take hours.

You can limit it to one accident using command line instructions:
* -a: String of accident ID, 'A-ID'. Eg 'A-809'
* -c: String of csv containing accident database. If you want to manually modify the 'US_Accidents_June20.csv' and rename it, this is where you would put the new title. But be careful, the code was written assuming the 'US_Accidents_June20.csv' is being loaded in.   
* -d: int representing the size of the geographic window around the accident. default is 250m which gives a 500m x 500m window since distance is from center to edge
* -g: int representing grid size for the danger matrix. 

Example Command Line Input (would recommend trying first):

```
python data_process.py -a 'A-809'
``` 



