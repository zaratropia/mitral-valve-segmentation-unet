# mitral-valve-segmentation-unet
Preprocess and train scripts for complete mitral valve segmentation from ultrasound data

## Usage
#### Preprocess

Place previously extracted train data into the `raw\train` directory. Each .nrrd image needs a corresponding
mask with the same filename and a `_mask` suffix. For example train_image01.nrrd would need a mask file named 
train_image01_mask.nrrd. Otherwise the data process script cannot load the mask data.

To use the broken data folder place the broken mask files into subdirectories under the `raw/broken` directory.
The subdirectories need the class of data quality as folder name. For exmaple `raw/broken/6` has the broken
files with six seperate parts of the mitral valve. Also make sure `cleanup_broken_data()` is called in the 
main function before loading the data. 

The images for testing should be in the `raw/test` directory. After train a prediction for the mask on each image
is made. 

If the data is placed into the right folders, just run the `data.py` script with Python and watch
how the numpy files are generated (in case enough space is available). 

#### Train and predict

After the generation of the numpy files just run the `train.py` script. Training will start, default 20 epochs.
This parameter can be easily adjusted in the code. Just check the parameters of the `model.fit` function.

The trained weights are used after the training to predict masks on the loaded test files. The output includes .nrrd and .tiff files for more convenient data access. You can find them in the `preds` directory.


#### Directory structure for train and test data
>>>>>>> Add some details to readme

raw/  
├── train  
├── test  
└── broken  
