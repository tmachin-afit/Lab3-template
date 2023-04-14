import datetime
import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
import statsmodels.api as sm
import tensorflow as tf
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from statsmodels.formula.api import ols
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import RMSprop
from tqdm.keras import TqdmCallback
# example of imports used by the instructor, you can ignore using these if you like


def new_york_city_airbnb():
    # Step 1 Read in the raw data
    # data is from https://www.kaggle.com/dgomonov/new-york-city-airbnb-open-data you must make an account first
    df = pd.read_csv('/opt/data/AB_NYC_2019.csv')
    print(df.columns)

    # baseline linear regression for comparison
    do_linear_regression = True
    if do_linear_regression:
        # make the model function
        mod = ols(
            f'price ~ C(neighbourhood_group) + C(room_type) + minimum_nights + availability_365 + number_of_reviews',
            data=df)
        # fit the model
        results = mod.fit()
        # print the fancy stats on our results
        print(results.summary())
        # plot one of the dimensions
        fig, ax = plt.subplots()
        sm.graphics.plot_fit(results=results, exog_idx="number_of_reviews", ax=ax)
        plt.show()

        # get the truth data and the prediction
        truth = df[['price']].to_numpy()[:, 0]
        pred = results.predict(df).to_numpy()

        # get the root mean square error
        rmse = np.sqrt(np.mean(np.square(truth - pred)))
        print(f"Linear Regression RMSE: {rmse}")
        
        # plot our truth vs. prediction
        # this should fit a 45 degree line if we were "perfect"
        plt.plot(truth, pred, '.', alpha=0.1)
        plt.xlabel('$ price truth')
        plt.ylabel('$ price pred')
        plt.title('Linear Regression Truth vs. Prediction')
        plot_min = 0
        plot_max = 400
        plt.plot([plot_min, plot_max], [plot_min, plot_max], 'k')
        plt.axis([plot_min, plot_max, plot_min, plot_max])
        plt.show()

    # do Steps 2-7 here

    # Print Final model performance metric
    # Final model performance metric is: XXX (write your metric in the comment)


def plot_cm(actual: np.ndarray, prediction: np.ndarray):
    """
    make a plot for a confusion matrix
    
    :param actual: the actual classes 
    :param prediction:  the predicted classes
    :return: None
    """
    # use the probabilities to make actual predictions of each class
    prediction[prediction > 0.5] = 1
    prediction[prediction <= 0.5] = 0
    # use pandas to make a confusion matrix
    data = {'y_Actual': actual.squeeze(),
            'y_Predicted': prediction.squeeze()
            }
    df = pd.DataFrame(data, columns=['y_Actual', 'y_Predicted'])
    confusion_matrix = pd.crosstab(df['y_Actual'], df['y_Predicted'], rownames=['Actual'], colnames=['Predicted'])
    # use seaborn to plot a heatmap of the confusion matrix
    sn.heatmap(confusion_matrix, annot=True)
    plt.show()


def titanic():
    # Step 1: read in the raw data
    # titanic_file_path = tf.keras.utils.get_file("/tmp/.keras/train.csv", "https://storage.googleapis.com/tf-datasets/titanic/train.csv")
    titanic_file_path = '/opt/data/train.csv'
    df = pd.read_csv(titanic_file_path)
    df = df.rename(columns={"class": 'class_'})
    print(df.columns)

    # baseline linear regression for comparison
    show_ols = True
    if show_ols:
        # make the model function
        mod = ols(
            f'survived ~ C(sex) + age + n_siblings_spouses + parch + fare + C(class_) + C(deck) + C(embark_town) + C(alone)',
            data=df)
        # fit the model
        results = mod.fit()
        # print the fancy stats on our results
        print(results.summary())
        # plot one of the dimensions
        fig, ax = plt.subplots()
        sm.graphics.plot_fit(results=results, exog_idx="age", ax=ax)
        plt.show()

        # get the truth and the predictions
        truth = df[['survived']].to_numpy()
        pred = results.predict(df).to_numpy()
        pred[pred > 0.5] = 1
        pred[pred <= 0.5] = 0
        
        # print a classification report from sklearn
        print(classification_report(y_true=truth, y_pred=pred, target_names=['died', 'survived']))
        
        # plot the confusion matrix
        plot_cm(truth, pred)

    # do steps 2-7 here

    # Print Final model performance metric
    # Final model performance metric is: XXX (write your metric in the comment)


def main():
    new_york_city_airbnb()
    titanic()


if __name__ == '__main__':
    main()
