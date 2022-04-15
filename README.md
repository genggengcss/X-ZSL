# X-ZSL
Code and data for the paper "[**Explainable Zero-shot Learning via Attentive Graph Convolutional Network and Knowledge Graphs**](https://content.iospress.com/articles/semantic-web/sw210435)".
Yuxia Geng, Jiaoyan Chen, Zhiquan Ye, Huajun Chen and others.
A research article published in Semantic Web Journal vol. 12, no. 5, pp. 741-765, 2021.


We implement the AZSL with two state-of-the-art GCN-based ZSL models -- [GCNZ](https://arxiv.org/abs/1803.08035) and [DGP](https://arxiv.org/abs/1805.11724),
i.e., **AZSL-G** and **AZSL-D**.

### Requirements
* `python 2`
* AZSL-G is developed with `TensorFlow 1.4`, and AZSL-D is developed with `PyTorch 1.2`  

### Dataset Preparation
#### Images of Unseen Class
We test the model on two datasets -- AwA and ImageNet animal subset (i.e., ImNet_A).  
 
**AwA**: Download [AwA](http://cvml.ist.ac.at/AwA2/AwA2-data.zip) (13GB!) and uncompress it to the folder `'data/images/'`. 
Note that we rename the awa class to its wordnet ID for conveniently training and testing.   
```
python data/process_awa.py
```
**ImNet_A**: The original images of ImageNet can be downloaded from [image-net.org](http://image-net.org/download-images), you need to register a pair of username and access key to acquire
and put them to the folder `'data/images/ImNet_A/'`.  
```
python data/download_ImNet.py --user $YOUR_USER_NAME --key $YOUR_ACCESS_KEY
```
* Note that all images of ImageNet take about 1.1 T, we only download the unseen classes we tested.

#### Word Embeddings of Class
You can skip this step if you just want to use the AZSL model we trained.

We use word emebddings of class names to initialize the graph nodes, these embedddings are trained using Glove model.  
You need to download the pretrained word embedding dictionary from
[here](http://nlp.stanford.edu/data/glove.6B.zip) and put it to the folder `'data/'`, and produce the word emebddings of class names yourself.
The scripts are provided with two AZSL models.

### Performance of AZSL
We introduce the training process of AZSL-G and AZSL-D in the corresponding directories.
Please read [AZSL-G/readme.md](/AZSL-G/readme.md) and [AZSL-D/readme.md](/AZSL-D/readme.md).  

We report the results of AZSL on ImNet_A compared with GCNZ and DGP.
Other results are shown in the paper.

|Method|Hit@1|Hit@2|Hit@5|
|----|-----|----|-----|
|GCNZ|29.31|47.11|71.63|
|AZSL-G|30.57|48.23|71.32|
|DGP|34.47|51.59|74.79|
|AZSL-D|34.81|51.72|74.54|

### Impressive Seen Class
With the attention weights learned from AZSL and the threshold, 
we can get the impressive seen classes for each unseen class.

The scripts are listed in the corresponding directories. 
We provide the results of AZSL-G in the file `data/X_ZSL/IMSC.json`, based on which the explanations are generated.

### Explanation Generation
We introduce the procedure of explanation generation in detail in the directory `X_ZSL`, please read [X_ZSL/readme.md](/X_ZSL/readme.md).

### How to Cite
If you find this code useful, please consider citing the following paper.
```bigquery
@inproceedings{geng2021explainable,
  author    = {Yuxia Geng and
               Jiaoyan Chen and
               Zhiquan Ye and
               Zonggang Yuan and
               Wei Zhang and
               Huajun Chen},
  title     = {Explainable zero-shot learning via attentive graph convolutional network
               and knowledge graphs},
  journal   = {Semantic Web},
  volume    = {12},
  number    = {5},
  pages     = {741--765},
  year      = {2021},
  url       = {https://doi.org/10.3233/SW-210435},
  doi       = {10.3233/SW-210435},
  timestamp = {Mon, 20 Sep 2021 08:52:07 +0200},
  biburl    = {https://dblp.org/rec/journals/semweb/GengCYYZC21.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```
