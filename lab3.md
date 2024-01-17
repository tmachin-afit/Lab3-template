# Lab 3
In this lab we will practice the universal workflow of machine learning. You will practice each step and demonstrate proficiency through commented code and plots as directed. There will be two datasets in this lab to show both classification and regression problems.

## New York City Airbnb Open Data
Our first problem will be to determine the price of a room in NYC. The dataset we will be using is the `New York City Airbnb Open Data` dataset available on kaggle [here](https://www.kaggle.com/dgomonov/new-york-city-airbnb-open-data). For reference, a linear regression model was able to achieve a mean absolute error (MAE) of $75 across the entire dataset. You may be asked to create a Kaggle account to download the data. This is free, and recommended for accessing other data sets (such as your final project data if necessary). 

## Titanic Survivors
Our second problem will be to determine if a passenger of the Titanic survived or not. The dataset we will be using is from Google examples and is available [here](https://storage.googleapis.com/tf-datasets/titanic/train.csv). For reference, a linear regression was able to achieve an accuracy of about 81% on the entire dataset. 

### Template
The template already has code to load these datasets, please do not change any paths in the code and instead download/move any data to the spot given by the code. If you modify the paths then it will make it much harder to grade, since the professor will have to inspect your code and change the paths back in order to run it.

# Steps
Each of these steps will be done for both problems.

## Step 1 Problem Statement and assembling dataset
You will load the raw data and split into input and output data. This split of the raw data will depend on what you think will be best to solve this problem (the inputs) and what the goal of the problem is (the output). 

### Deliverable
In your code make a comment to mark the section where you load your raw data and then split it. Mark this block of code as step 1.

## Step 2 Measures of Success
Determine your own metric based on the problem. Remember we are determining the price of the room and if someone survived on the Titanic or not.

### Deliverable
In your code comment where you choose a measure of success.


## Step 3 Deciding an Evaluation Protocol
Choose your own method of evaluation of the dataset. This means picking a train/validation/test split of the data. 

### Deliverable
Comment in code where you made you own train/validation split and where you have the hold out test set. 

## Step 4 Preparing your data
In this step you will prepare your data for the ANN. Remember ANNs require numbers and our raw data may not always be numbers or numbers that aren't suited for ANNs. Students will comment in code exactly what they are doing in order to process this raw data into numbers that ANNs can use.

### Deliverable
Converting the data will be a multi step process. Use as many comments as possible to document what you are doing to get a final data set to train the ANN with.

## Step 5 Develop a Model and Determine a Baseline
The real work in this step is to make an ANN which compiles. At first, you should not stress about the performance when building the network we only want an ANN that can process the input to the output and evaluate our loss function and metrics.

## Deliverable
In your code leave this network in place and mark it with a comment. It does not have to run in the final code and can be inside some if statement that does not run in the final code. 

## Step 6 Scaling Up
Scale up the ANN so that it overfits to the training data. Remember the goal here is to figure out how much capacity is required for your dataset. The datasets in this lab are very small so scaling up may only be slightly larger compared to the initial network.

## Deliverable
Just like the previous step keep this network (even if it is inside an if statement and not used) and mark it with a comment. 

## Step 7 Generalizing your Model
In the previous section at one point the training and validation performance may have diverged, meaning the training performance got better while the validation performance got worse. Our goal here is to add regularization techniques until our validation performance is closer to the training performance and ideally our validation performance reaches a pre-determined measure (ideally beats our baseline but this may not always be true).

After you are satisfied with your validation performance you then evaluate against your test set and publish (submit the lab).

## Deliverable
Mark this network in your code however it should run in the final submission. 
Also included should be the final performance you achieved on the test set. You should print this final performance metric and also put your final performance in a comment (in case it changes slightly when I run the code).

## Easier to Grade
To grade this lab I will clone your repo and then in the project root run the command
```bash
python3 lab3.py
```
That should be enough to run the entire lab and show me all the output I have asked for except for the commented networks which I will look at the code directly for.
