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
* Update feature name in makefile for feature generation in `Makefile.feature.$FEATURE_NAME`
# Training a model
* Add code to setup training of an algorithm in `src/train_predict_$ALGO_NAME.py`
* Change name of algorithm (under ALGO_NAME) in `Makefile.$ALGO_NAME` to $ALGO_NAME
* Also in `Makefile.$ALGO_NAME`, "include" the makefile for the feature you want to use to train the model
    * For example, add `include Makefile.feature.j1` at the top of `Makefile.lgb` to train the LightGBM model using features created by `generate_j1.py`
# Training a meta-classifier with stacking
* List the names of all the base models you want to stack in `Makefile.feature.esb1` under BASE_MODELS
* Give a name to you ensemble under FEATURE_NAME in `Makefile.feature.esb1`
* Create a model makefile, say `Makefile.lgb1` and add `include Makefile.feature.esb1` to the top of the makefile.
    * Make sure only `Makefile.feature.esb1` is included in `Makefile.lgb1`
* Run `make -f Makefile.lgb1` to train the meta-classifier
# Making a submission
* Use the Kaggle API to make a submission. For example,
```
kaggle competitions submit -c cat-in-the-dat-ii -f build/sub/lgb1_esb1.sub.csv -m "CV: 0.755609"
```
* CV scores can be found in `build/metric`

