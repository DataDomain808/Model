# Solubility prediction of BCS class II drugs through combining machine learning and molecular descriptor


> ![TOC](https://i.postimg.cc/HLL5XYpX/toc.png)

This repository consists of 1 excel file and 13 python scripts: 

1. excel file

    Sheet1: dataset used to train the model for subsequent analysis 

    Sheet2: dataset used to construct the learning curves 

    Sheet3: dataset used for the uncommon solvent prediction 

    Simple: dataset with SMR_VSA1 and Chi0 molecular descriptors only

2. python script

    SVM model.py: constructed support vector machine model

    KNN model.py: constructed K-nearest neighbor model

    DT model.py: constructed decision trees model

    RF model.py: constructed random forest model

    ANN model.py: constructed artificial neural network model

    XGBoost model.py: constructed extreme gradient boosting model

    MLR model.py: constructed multiple linear regression model

    Learning curve.py: Learning curve of the dataset with the XGBoost model

    CS.py: After extracting the molecular descriptors of BCS class II drugs by RDkit, the descriptors were preprocessed with cosine similarity

    SPCA.py: After extracting the molecular descriptors of the solvents by RDkit, the descriptors were preprocessed with sparse principal component analysis

    Uncommon solvent predictions.py: Uncommon solvent prediction with the XGBoost model

    SHAP analysis.py: Shapley Additive Explanations analysis with the XGBoost model

    Feature Importance.py: feature importance analysis with the XGBoost model

## Installation
Dependencies: ```scikit-learn==1.5.0```, ```scikit-optimize==0.10.2```, ```xgboost==2.1.4```, ```rdkit=2023.09.5``` 


