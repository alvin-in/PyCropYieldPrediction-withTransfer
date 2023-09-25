# PyCropYieldPrediction-withTransfer
An extension of the [Gabriel Tseng's](https://github.com/gabrieltseng/pycrop-yield-prediction) PyTorch implementation of 
[Jiaxuan You's](https://cs.stanford.edu/~ermon/papers/cropyield_AAAI17.pdf) Deep Gaussian Process with CNN for soybean crop forecasting in Argentina.
In addition, code components from the work ["Deep Transfer Learning for Crop Yield Prediction with Remote Sensing Data"](https://github.com/AnnaXWang/deep-transfer-learning-crop-prediction)
are used to export Argentine satellite data. 

# Pipeline

# Setup
To set up the environment, the package manager Anaconda with Python 3.7 is required. 
Run
  ```sh
  conda env create -f crop_yield_prediction.yml
  ```
to create an environment named `crop_yield_prediction` and run
  ```sh
  conda activate crop_yield_prediction
  ```
to activate the envoironment. Additionally you need to sign up to [Google Earth Engine](https://developers.google.com/earth-engine/)
and authenticate yourself within the `crop_yield_prediction` environment by runnning
  ```sh
  earthengine authenticate
  ```
and following the instructions. 
