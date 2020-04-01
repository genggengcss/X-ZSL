# AZSL_D
The code is for AZSL based on DGP.


### Requirements
* python 2
* PyTorch 1.2  


### Data Preprocess
We start with ImNet_A dataset, AwA is processed by specifying the parameter `--dataset AwA`. 

#### Extract CNN Features of Images
During testing, the model first uses pre-trained CNN model (e.g., ResNet) to extract the features of testing images, and then conduct nearest neighbor classification.  

1. Download pre-trained CNN model ([ResNet-50](https://download.pytorch.org/models/resnet50-19c8e357.pth) implemented with PyTorch)
and put it to the folder `'data/AZSL-D/materials/'`.

2. Extract ResNet base model and fc-weights.
```
python src/process_resnet.py
```
3. Extract features of unseen images and save them to the corresponding dataset directory (`'data/AZSL-D/*/Test_DATA_feats/'`).
```
python src/extract_img_feats.py
```

#### Prepare AGCN Training Data

* Build Graph and Initialize graph nodes (prepare pre-trained Glove embeddings in advance).
```
python src/make_induced_graph.py
```

### Training and Testing

```
python train_predict_adgp.py
python test_adgp.py --pred 680
```
* You can run the testing commands with the predicted model we trained (the file `'epoch-680.pred'` saved [here]()).
* Note that `train_predict_dgp.py` and `test_dgp.py` is the original implementation of DGP. 
* `test_weights.py` is used for selecting the impressive seen classes for each unseen class using learned attention weights.