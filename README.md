# crash_net
Predict traffic accidents from an image of the road network

## Quick Summary
This is the initial commit of my convolutional neural network built using keras. 

The python file data_process.py will generate the input image and output matrix that train.py will use to train.

A long writeup of this proposed project can be found here: (https://www.reubenbritto.com/tdi-proposal)

## The data I'm processing

The data I'm processing comes from two sources: a database (https://smoosavi.org/datasets/us_accidents) of 3.5 million US accidents recorded from February 2016 to June 2020, and the osmnx python package (https://geoffboeing.com/2016/11/osmnx-python-street-networks/) that can generate node/edge graphs of the road network anywhere in the US.


## How to run the code

First, I recommend creating two seprate virtual environments using condas (the creator of osmnx recommends condas specifically). One virtual environment to run data_process.py, and another virutal environment to run train.py  

### Running data_process.py to generate training examples.

Once you've installed condas, run the code below to create an (ox) virtual environment.

```
conda config --prepend channels conda-forge
conda create -n ox --strict-channel-priority osmnx
```

To run the data_process.py script, you'll need three things: 
* Accidents_by_zip.csv: this file is provided when you clone the repo. It's a tally of the number of accidents by zip code
* US_Accidents_June20.csv: this file you can download from [here](https://osmnx.readthedocs.io/) (this is the database from smoosavi referenced above)
* model_config.json: this file specifies which geographical locations to randomly pull training examples, and how many training examples to pull from US_Accidents_June20.csv to generate your training set.

Once you have the above three things, if you run the following:
 
```
python data_process.py
```

the code will start generating input and output images for all the accidents that have occurred in SF and put them in datasets/[dataset_name]/ that you specify in the "date_process" part of the model_config.json file. This will take hours. You can and should use config file to limit where and how many accidents are generated.

### Running train.py to train a CNN.

Run the code below to create an (ml) virtual environment for running keras.

```
conda create -n ml
conda install -r requirements.txt
```

You can now run 

```
python train.py
```

but you should customize the "model" part of the model_config.json first. In this config file, you can actually specify the architecture of your CNN, and all the hyperparameters. Once run, the model outputs will be placed into a new folder within the datasets/[dataset_name] folder.



