# PyCropYieldPrediction-withTransfer
An extension of [Gabriel Tseng's](https://github.com/gabrieltseng/pycrop-yield-prediction) PyTorch implementation of 
[Jiaxuan You's](https://cs.stanford.edu/~ermon/papers/cropyield_AAAI17.pdf) Deep Gaussian Process built on a CNN for soybean crop forecasting in Argentina.
In addition, code components from the work ["Deep Transfer Learning for Crop Yield Prediction with Remote Sensing Data"](https://github.com/AnnaXWang/deep-transfer-learning-crop-prediction)
are used to export Argentine satellite data.

The code was used to produce the results published in the publication "Leveraging Remote Sensing Data for Yield Prediction with Deep Transfer Learning". The document is open access and can be found at https://www.mdpi.com/1424-8220/24/3/770 . If you find our code helpful, please cite our work as follows:

@Article{s24030770,
AUTHOR = {Huber, Florian and Inderka, Alvin and Steinhage, Volker},
TITLE = {Leveraging Remote Sensing Data for Yield Prediction with Deep Transfer Learning},
JOURNAL = {Sensors},
VOLUME = {24},
YEAR = {2024},
NUMBER = {3},
ARTICLE-NUMBER = {770},
URL = {https://www.mdpi.com/1424-8220/24/3/770},
ISSN = {1424-8220},
DOI = {10.3390/s24030770}
}

# Pipeline
## USA
### Exporting
Run 
  ```sh
  python run.py export
  ```
to export the US satellite data into your Google Drive. You will need up to 165 Gb of storage. The export class allows checkpointing.
The [Earth Engine Task Manager](https://code.earthengine.google.com/tasks) shows your ongoing tasks. This may take longer. 
Once all the data has been exported to your Google Drive, you can drag the folders `crop_yield-data_image`, `crop_yield-data_mask` and 
`crop_yield-data_temperature` into your local data folder ([Google Drive Desktop](https://www.google.com/intl/en/drive/download/) is recommended, 
otherwise the data will be downloaded in a lot of ZIP files). 

### (Optional) Data Cleansing
Our cleaned data are given in the data folder. If you want to use our data cleansing (>2000 cropland pixel) on your own data, you have to run 
  ```sh
  python run.py data_cleansing
  ```
and 
  ```sh
  python cyp/data/merge_yield_pix-count_usa.py
  ```
Note here that the corresponding csv are addressed according to their column orders.

###  Preprocessing
  ```sh
  python run.py process
  ```
Merges data and splits them by year. Saves files as `.npy` files.

### Feature Engineering
  ```sh
  python run.py engineer
  ```
Generates histograms from the processed `.npy` files.

### (Optional) Hyperparameter tuning
  ```sh
  python run.py run_optuna_usa
  ```
Non cross-validated hyperparameter search (run `hyp_multi_trans_cnn_usa` for a ten-fold cross validation, but it's runtime is immense). 
Results are saved in the `data` folder with the name given by `out_hyp_csv`.

### Model Training
  ```sh
  python run.py train_cnn
  ```
Trains the CNN and saves the model and the results in `data/models/<new_model>`. Additional information are saved into your [Weights and Biases](https://wandb.ai/site) account.

## Argentina
The basic procedure in Argentina is the same, but in some places paths or names need to be adjusted. The descriptions can be taken from the US Pipeline and are not repeated here.
### Exporting
  ```sh
  python cyp/data/argentina_export.py
  ```
### (Optional) Data Cleansing
  ```sh
  python run.py data_cleansing
  ```
Adjust the names and paths inside run.py to the Argentinian values as it is commented.
  ```sh
  python cyp/data/yield-csv_to-utf_with-buxacre.py
  ```
This removes Spanish characters, converts tons per acre to bushels per acre, and applies data cleansing of at least 2000 cropland pixels. 
The variable `YIELDFILE` in the head of the script can be changed to the name of your yield data file.
### Preprocessing
  ```sh
  python run.py process_argentina
  ```
### Feature Engineering
   ```sh
  python run.py arg_engineer
  ```
### (Optional) Hyperparameter tuning
  ```sh
  python run.py run_optuna
  ```
### Model Training
  ```sh
  python run.py train_trans_cnn
  ```
To change the referenced US Model, the paths within models/transfer_base.py and models/transfer_convnet.py must be adjusted.

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
to activate the environment. <br>
Additionally you need to sign up to [Google Earth Engine](https://developers.google.com/earth-engine/)
and authenticate yourself within the `crop_yield_prediction` environment by runnning
  ```sh
  earthengine authenticate
  ```
and following the instructions. <br>
[Weights and Biases](https://wandb.ai/site) is used to track experiments. Run
  ```sh
  wandb login
  ```
and follow the instructions to activate wandb. You can also disable it by running
  ```sh
  wandb disabled
  ```
