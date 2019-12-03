# Text_GCN
An reimplementation of Graph Convolutional Networks for text classification using tensorflow

The referenced code can be found [here](https://github.com/yao8839836/text_gcn).

Liang Yao, Chengsheng Mao, Yuan Luo. "Graph Convolutional Networks for Text Classification." In 33rd AAAI Conference on Artificial Intelligence (AAAI-19), 7370-7377

## Requirements

python 3.6  
  
Tensorflow 1.14


## Data pre-processing and training network
Example:

    $ cd ./preprocess
    $ python remove_words.py <dataset>
    $ python build_graph.py <dataset>
    
    $ cd ..
    $ python main.py --dataset <dataset>
