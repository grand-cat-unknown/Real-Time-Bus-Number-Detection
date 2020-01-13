                                     
                                                          


# Real-time Bus Number Detection

Implemented a bus number detection in Tensorflow and Keras. Used Tiny-YOLOv2 and Tensorflow's Object Detection API model for the detection pipeline. Please read Report.pdf for understanding the design and package choices.

----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------

#Prerequistes:

It is highly suggested to use a virtual environment for this project.

For getting into Virtual Environment:

pip install virtualenv
virtualenv bus-numbers
source bus-numbers/bin/activate


This project requires Tensorflow Object Detection API. Check if you already have it.
	>> python
	>> import object_detection

If you don't have it. Run the following for its installation:

1. 
	git clone --depth 1 https://github.com/tensorflow/models

2.
	pip install tensorflow==1.15.0
3.
	sudo apt-get install protobuf-compiler python-pil python-lxml python-tk
	pip install --user Cython
	pip install --user contextlib2
	pip install --user pillow
	pip install --user lxml
	pip install --user jupyter
	pip install --user matplotlib

Now, check if protobuf-compiler is working:

	
	# GO TO THIS DIR: tensorflow/models/research/
	protoc object_detection/protos/*.proto --python_out=.


Sometimes "sudo apt-get install protobuf-compiler" will install Protobuf 3+ versions for you and some users have issues when using 3.5. If that is your case, try the manual installation:

https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md#Manual-protobuf-compiler-installation-and-usage


RECHECK IF THIS IS WORKING:

	# From tensorflow/models/research/
	protoc object_detection/protos/*.proto --python_out=.

IF YES:
	# From tensorflow/models/research/

	export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim

	Add it to ~/.bashrc.


If everything is set correctly. This line shouldn't throw any errors:

# From tensorflow/models/research/

python object_detection/builders/model_builder_test.py


----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------

#Dependencies:

I have implemented all of this code in Google Colab platform, which already has a lot of packages installed. So it has hard to extract the exact dependencies.
To the best of my knowledge requirements.txt contains all the required dependencies. 
You can install them by

pip install -r min-requirements.txt

xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
I have also added an exhaustive list of requirements, just in case. Please don't run this unless nothing else works. 
pip install -r exhaustive-requirements.txt
xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------

# FILES USAGE:

If Everything is installed correctly. You can use the following to detect and track Bus Numbers in a video

> python Bus_Number_Detector.py /path/to/video

The same code is available in Notebook format which is far more readable: ' Bus_Number_Detector.ipynb '

The notebook ' Multi_Digit_classifier.ipynb ' contains a TensorFlow implementation of a Multi-Digit classifier.


PLEASE READ REPORT.PDF for Design details

----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------


##FOLDERS DESCRIPTION

Bus_Detection: Contains model and data required to detect buses in video
|	
|->	model_data --> Used for loading the bus detection model. Used in 'Bus_Number_Detector.ipynb'
|->	yad2k 	   --> Used to convert Darknet model to Keras. 
|->	utils      --> Helper files for loading and processing the output of the bus detection model. Used in 'Bus_Number_Detector.ipynb'



Digit_Detection: Contains model and data required to detect digits in video
 |
 |-> Training MobileNet:  Contains a MobileNet model trained to DETECT AND CLASSIFY digits, trained on SVHN dataset
 |		|
 |		|->	Pre-process.ipynb 		--> Converts SVHN dataset to usable CSV format
 |		|->	Csv_to_Inference.ipynb  --> Uses CSV to train the Digit Detection model and creates an Inference graph
 |
 |-> TF_Implementation:	Contains a Tensorflow CNN trained to CLASSIFY digits, trained on SVHN dataset.
 |		|	
 |		|->	SVHN-preprocess.ipynb	--> Converts the SVNH dataset into h5 format for the classifier
 |		
 |-> Output_Graph2: 
			Contains the trained MobileNet on SVHN, being used in  ' Bus_Number_Detector.ipynb '

output_examples: Contains a few results of the Detection model on Training video samples.

