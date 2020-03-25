# AZSL_G
The code is for AZSL based on DGP.


### Requirements
* python 2
* PyTorch 1.2  


### Dataset Process
We start with ImNet_A dataset, AwA is processes by specifying the parameter `--dataset AwA`. 

#### Extract CNN Features of Images
During testing, the model first uses pre-trained CNN model (e.g., ResNet) to extract the features of testing images, and then conduct nearest neighbor classification.  

1. Download pre-trained CNN model ([ResNet-50](https://download.pytorch.org/models/resnet50-19c8e357.pth) implemented with PyTorch)
and put it to the folder 'data/AZSL-D/materials/'.

2. Extract base model and fc-weights
```
python src/process_resnet.py
```
3. Extract feature.
Extract features of unseen images and save them to the corresponding dataset directory ('data/AZSL-D/*/Test_DATA_feats').
```
python src/extract_img_feats.py
```

#### Prepare AGCN Training Data

* Build Graph and Initialize graph nodes .
```
python src/make_induced_graph.py
```

### Training and Testing

```
python train_predict_adgp.py
python test_adgp.py
```
* Note that `train_predict_dgp.py` and `test_dgp.py` is the original implementation of DGP. 

