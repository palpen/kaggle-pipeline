# kaggle-pipeline
Pipeline for data science competitions based on https://github.com/jeongyoonlee/kaggler-template

# Getting started
* Get the name of the competition
    * `kaggle competitions list`
* Update competition name in `Makefile`
* Download data
    * `make -f Makefile data`
# Feature engineering
* Add code to generate featuers in `src/generate_$FEATURE_NAME.py`
* Update feature name in makefile for feature generation in Makefile.feature.$FEATURE_NAME
# Training a model
* Add code to setup training of an algorithm in `src/train_predict_$ALGO_NAME.py
* Change name of algorithm (under ALGO_NAME) in `Makefile.$ALGO_NAME` to $ALGO_NAME
* Also in `Makefile.$ALGO_NAME`, "include" the makefile for the feature you want to use to train the model
    * For example, you can add `include Makefile.feature.j1` at the top of `Makefile.lgb` to train the LightGBM model using features created by `generate_j1.py`
# Training a meta-classifier with stacking
* 
