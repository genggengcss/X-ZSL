# AZSL_G
The code is for AZSL based on GCNZ.


### Requirements
* python 2
* TensorFlow 1.4

### Dataset Process
We start with ImNet_A dataset, AwA is processes by specifying the parameter `--dataset AwA`. 

#### Extract CNN Features of Images
During testing, the model first uses pre-trained CNN model (e.g., ResNet) to extract the features of testing images, and then conduct nearest neighbor classification.  

1. Download pre-trained CNN model ([ResNet-50](http://download.tensorflow.org/models/resnet_v1_50_2016_08_28.tar.gz) implemented with TensorFlow)
and put it to the folder `'data/AZSL-G/materials/'`.
2. Extract feature.
* Prepare image list and make labels.
We provide the list (`'test_img_list.txt'`) in the corresponding dataset directory (`'data/AZSL-G/*/'`).
```
python src/pre_test_img_list.py
```
* Extract features of unseen images and save them to the corresponding dataset directory (`'data/AZSL-G/*/Test_DATA_feats'`).
```
python src/extract_img_feats.py
```

#### Prepare AGCN Training Data

1. Build Graph with animal subset.
```
python src/io_graph.py
```
2. Initialize Graph nodes with word embeddings (prepare pre-trained Glove embeddings in advance).
```
python src/io_embed.py
```
3. Label Graph (input and output).
```
python src/io_train_sample.py
```

### Training and Testing

```
python train_predict_agcn.py
python test_agcn.py
```
* Note that `train_predict_gcn.py` and `test_gcn.py` is the original implementation of GCNZ. 



