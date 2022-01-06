# Ficc AI corp internal python package
### This repository contains a python package for internal development and research.

This package contains:
- A means to access various models by name and version numbers. Largely under the ```ficc.models``` sub-module.
- A means to query/cache data from the server and process it for training and/or evaluation. Largely under the ```ficc.data``` sub-module.
- A variety of utility functions to aide in the training and evaluation process. Largely under the ```ficc.utils``` sub-module.


### Installation
```pip install . [-upgrade]```

### Requirements
<ul> 
<li> pandas
<li> numpy
<li> tensorflow 
<li> keras-tuner
<li> pandarallel
<li> python gcloud api
</ul>

### API

#### Data package
The main driver for the ficc data processing package can be imported as folows ```from ficc.data.process_data import process_data``` 

The process data method takes 6 required and one optional parameter

- ```Query```  A query that will be used to fetch data from BigQuery.
- ```BigQuery Client```
- ```SEQUENCE_LENGTH``` the sequence length of the trade history can take 32 as its maximum value. 
- ``` NUM_FEATURES``` The number of features that the trade history contains.
- File path to save the raw data grabbed from BigQuery. 
- The yield curve to use acceptable options ```S&P``` or ```ficc```
- ```training_features``` A list containing the features that will be used for training. This is an optional parameter


### Example
An example of each API is available [here](https://github.com/Ficc-ai/ficc_python/blob/main/example.py)