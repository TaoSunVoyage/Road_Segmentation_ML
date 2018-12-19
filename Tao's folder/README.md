# Road Segmentation
EPFL CS-433 Machine Learning - Project 2

**`Team Members`:** Tao Sun, Xiao Zhou, Jimin Wang


## Instructions

In order to reproduce the result we submitted to crowdAI, please follow the instructions as following:

* Please make sure ```Python 3.6``` and packages in ```requirements.txt``` are installed.

~~~~shell
pip install -r requirements.txt
~~~~

* If you want to get the results we submitted in the crowdAI, we have prepared models for you. Please download it from [here](https://drive.google.com/uc?export=download&id=1xgT7HBzAAjZbNYnJzIx56ZidNGiWsLt0), then unzip and move the three weights file into ```weights\``` folder. Then run ```run.py```, you will get the ```submission.csv``` in the  ```submission\``` folder.

~~~~shell
python run.py
~~~~

* If you want to retrain our models, please first kindly download dataset from [crowdAI](https://www.crowdai.org/challenges/epfl-ml-road-segmentation/dataset_files), and then upzip and move ```training``` and ```test_set_images``` folders under the ```data\``` folder. Then run ```train_predict.py```, you will get the ```submission.csv``` in the  ```submission\``` folder.

~~~~shell
python train_predict.py
~~~~

## Modules

### ```data.py```

Functions for preprocessing data, building data generator and saving data.

### ```losses.py```, 	```metrics.py```

Define: Dice coefficient, Dice loss, BinaryCrossentropy+Dice loss and F1 score.

### ```model.py```

* 	`unet`: Modified U-Net with dropout and batch-normalization
*  `unet_dilated`: U-Net with dilated convolution as bottlenect


### ```mask_to_submission.py```

Helper functions for generating crowdAI submission file.

### ```train_predict.py```

Script to retrain models and get the prediction results as we submitted in crowdAI.

### ```run.py```

Script to generate the same submission file as we submitted in crowdAI with pretrained models.