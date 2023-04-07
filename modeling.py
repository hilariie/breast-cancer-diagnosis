"""
Modeling.py
-----------
This script holds relevant functions for creating, training and evaluating models
"""
import tensorflow as tf
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, KFold
from keras.regularizers import l2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, matthews_corrcoef, classification_report
from sklearn import svm


def train_models(xt, xtt, yt, ytt, model_list):
    """
    trains and evaluates models in a list

    Parameters
    ----------
        xt: dataframe, independent variables for training
        xtt: dataframe, independent variables for testing
        yt: dataframe, target variables for training
        ytt: dataframe, target variables for testing
        model_list: list, list of models to train and evaluate
    """
    for model in model_list:
        model.fit(xt, yt)
        model_pred = model.predict(xtt)
        if model == xgb:
            print('Model: XGBClassifier')
        else:
            print(f'Model: {model}')
        print(f"accuracy on train data: {model.score(xt, yt)}")
        print(f"accuracy on test data: {model.score(xtt, ytt)}")
        print(f'f1 score: {f1_score(ytt, model_pred)}')
        print(f'matthews coef score: {matthews_corrcoef(ytt, model_pred)}\n')

def nn_model(x_train, blocks, dropout, regu, final, activation, first):
    """
        Creates a Neural network model with specified architecture

        Parameters:
        -----------
        x_train: numpy array
            Data the model is trained on - used to get the number of columns.
        blocks: list
            list of integers representing the number of neurons in each hidden layer.
        dropout: float
            Dropout rate, a float value between 0 and 1, to be applied the hidden layers.
        regu: float
            Regularization strength, a float value controlling the strength of L2 regularization applied to hidden layers.
        final: int
            number of neurons in the output layer
        activation: string
            Activation function for the output layer
        first: int
            Number of neurons in the first hidden layer.

        Returns:
        --------
        model: Keras Sequential model
            The created neural network model with the specified architecture.
        """
    model = Sequential()
    model.add(Dense(first, activation='relu', input_dim=x_train.shape[1]))
    for i in blocks:
        model.add(Dense(i, activation='relu', kernel_regularizer=l2(regu)))
        model.add(Dropout(dropout))
    model.add(Dense(final, activation=activation))
    return model

def nn_compile(x_train,
               x_val,
               y_train,
               y_val,
               loss,
               optimizer,
               epochs,
               batch_size,
               blocks,
               dropout,
               regu,
               activation,
               patience,
               first,
               verbose,
               monitor):
    """
    Compiles and trains a neural network model using Keras Sequential API with specified parameters.

    Parameters:
    ----------
    x_train: pandas DataFrame
        Training data input with shape (number of samples, number of features).
    x_val: pandas DataFrame
        Validation data input with shape (number of samples, number of features).
    y_train: pandas Series
        Training data output with shape (number of samples,).
    y_val: pandas Series
        Validation data output with shape (number of samples,).
    loss: string or Keras loss function
        Loss function to be used during model training.
    optimizer: string or Keras optimizer object
        Optimization algorithm to be used during model training.
    epochs: int
        Number of times the entire dataset will be passed through the model during training.
    batch_size: int
        Number of samples to be used in each update of the model weights during training.
    blocks: list
        List of integers representing the number of neurons in each hidden layer of the neural network model.
    dropout: float
        Dropout rate, a float value between 0 and 1, to be applied to the hidden layers of the neural network model.
    regu: float
        Regularization strength, a float value controlling the strength of L2 regularization applied to hidden layers of the neural network model.
    activation: string
        Activation function for the output layer of the neural network model.
    patience: int
        Number of epochs to wait before early stopping during model training.
    first: int
        Number of neurons in the first hidden layer of the neural network model.
    verbose: int
        Verbosity level during model training (0 = silent, 1 = progress bar, 2 = one line per epoch).
    monitor: string
        Metric to be monitored for early stopping during model training.

    Returns:
    --------
    hist: dict
        Dictionary containing the training history, including loss and accuracy, for each epoch.
    _model: Keras Sequential model
        The trained neural network model.
    """
    if activation == 'softmax':
        final = 2
    elif activation == 'sigmoid':
        final = 1
    # define and compile neural network model
    _model = nn_model(x_train, blocks=blocks, dropout=dropout, regu=regu, final=final, activation=activation, first=first)
    _model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
    # set callback clause/method
    callbacks = [tf.keras.callbacks.EarlyStopping(monitor=monitor, patience=patience)]
    # train the model
    hist = _model.fit(x_train.values,
                      y_train.values,
                      callbacks=callbacks,
                      validation_data=(x_val, y_val),
                      epochs=epochs, 
                      batch_size=batch_size,
                      verbose=verbose)
    return hist.history, _model


def nn_evaluation(model, 
                  x_train, 
                  x_val, 
                  x_test,
                  y_train,
                  y_val,
                  y_test, 
                  sparse=False):
    """
    Evaluate the performance of a neural network model on training, validation, and test data.

    Parameters:
    -----------
    model: tf.keras.Model
        The compiled neural network model.
    x_train: numpy.ndarray
        The input training data.
    x_val: numpy.ndarray)
        The input validation data.
    x_test: numpy.ndarray)
        The input test data.
    y_train: numpy.ndarray
        The target training data.
    y_val: numpy.ndarray)
        The target validation data.
    y_test: numpy.ndarray)
        The target test data.
    sparse: bool, optional)
        Whether the target data is sparse or not. Defaults to False.

    Returns:
        None
    """
    # get predictions on training, validation, and test data
    pred = model.predict(x_train, verbose=0)
    vpred = model.predict(x_val, verbose=0)
    tpred = model.predict(x_test, verbose=0)

    # convert the neural network predictions to a binary format (0 or 1)
    if sparse:
        pred = prd(pred)
        vpred = prd(vpred)
        tpred = prd(tpred)
    else:
        pred = [round(predictions[0]) for predictions in pred]
        vpred = [round(predictions[0]) for predictions in vpred]
        tpred = [round(predictions[0]) for predictions in tpred]

    # Evaluate the model's predictions
    print('Training accuracy')
    print(f'f1 score: {f1_score(y_train, pred)}')
    print(f'matthews coef score: {matthews_corrcoef(y_train, pred)}\n')

    print('Validation accuracy')
    print(f'f1 score: {f1_score(y_val, vpred)}')
    print(f'matthews coef score: {matthews_corrcoef(y_val, vpred)}\n')

    print('Test accuracy')
    print(f'f1 score: {f1_score(y_test, tpred)}')
    print(f'matthews coef score: {matthews_corrcoef(y_test, tpred)}\n')
    """
        Converts sparse predictions(list of lists) to binary format (list of 0s or 1s as elements)
        :param pred: model predictions :type: list
        :return new_pred: list of predictions as numbers for easy interpretation :type: list
        """
def prd(pred):
    """
    Convert neural network predictions to a binary format (0 or 1).

    Parameters:
    ----------
        pred (numpy.ndarray): The predictions made by the neural network.

    Returns:
    ---------
        list: The converted binary predictions.
    """
    new_pred = []
    for i in pred:
        i = np.round(i, 5)
        for ii_index, ii in enumerate(list(i)):
            if ii == max(i):
                new_pred.append(ii_index)
    return new_pred

def prd2(pred, threshold):
    new_pred = []
    for i in pred:
        i = np.round(i, 5)
        if i[0] < threshold:
            new_pred.append(1)
        else:
            new_pred.append(0)
    return new_pred

def classification_rep(model, x_Test, y_Test, plot=False, return_=False):
    """
    Generate a classification report for a trained model's predictions.

    Parameters:
    ----------
    model: keras.Model
        The trained neural network model.
    x_Test: numpy.ndarray
        The test data input for prediction.
    y_Test: numpy.ndarray
        The true labels for the test data.
    plot: bool, optional
        Whether to plot the confusion matrix. Default is False.
    return_: bool, optional
        Whether to return the predictions. Default is False.

    Returns:
    ---------
    pred: numpy.ndarray
        The predicted labels, if return_ is True.
    """
    pred = model.predict(x_Test)
    print(classification_report(y_Test, pred))
    if plot:
        confusion_plot(y_Test, pred)
    if return_:
        return pred

def plot_train_hist(history):
    """
    Plot the training history of a neural network model.

    Parameters:
    ----------
    history:(dict)
        The history object returned by model.fit() containing training loss, validation loss,
        training accuracy, and validation accuracy.
    """
    fig, ax = plt.subplots()
    ax.plot(history['loss'])
    ax.plot(history['val_loss'])
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(['training loss', 'validation loss'])
    plt.show()
    plt.plot(history['accuracy'])
    plt.plot(history['val_accuracy'])
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(['training accuracy', 'validation accuracy'])
    plt.show()

