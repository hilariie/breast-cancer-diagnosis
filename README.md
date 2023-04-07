# Breast Cancer diagnosis
The objective of this project is to develop an effective breast cancer diagnostic model using data mining techniques. The project employs a range of methods including data extraction, data pre-processing, statistical modeling, and deep learning architecture. In the data pre-processing stage, four different data samples were prepared and tested using four statistical algorithms, namely logistic regression, support vector machines, random forest, and xgboost. The best performing data sample was identified through these tests, and the best performing model on the best performing data sample was further evaluated. Given the crucial importance of recall in cancer diagnosis, the xgboost model was identified as the best as it was found to have the highest recall among the four statistical models. The xgboost model was then compared to a complex neural network model that initially had a low recall, however, after adjusting the threshold of the model, the recall significantly increased. The neural network model achieved a recall of 99% for malignant cases, thereby demonstrating its effectiveness as a diagnostic tool for breast cancer. However there are some trade-offs as the model with high recall had a relatively low precision compared to other models.

## File description
* breast_cancer.html - This file contains the jupyter notebook code in which the experiments were done.
* preprocessing.py - This file contains the neccessary functions for preprocessing the dataset
* modeling.py - This script contains the neccessary functions for creating, training and evaluating models.
* breast_cancer.py - This script is called to preprocess, train, and display evaluation of models. This script is configured via the `config.yaml` file.
* config.yaml - This file holds neccessary configurations to be made regarding the pre-processing steps and models (including parameters) to experiment.

## Dependacies
This program has been tested on Tensorflow for CPU and Tensorflow-GPU. To install required libraries, run the following command, which was generated using `pip freeze` on WSL.
```
pip install -r requirements.txt
```

## How to run
To run the project, execute the following command:

```
python breast_cancer.py
```

You can modify the `config.yaml` values to experiment with different results, such as adujusting model parameters or pre-processing steps.