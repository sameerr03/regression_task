# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 18:39:38 2024

@author: samvi
"""

import argparse
import pandas as pd
import numpy as np
import os
import pickle

from data_preprocessing import group_vehicle_classes, encode_vehicle_class, MinMaxScaler

class LinearRegression:
    def __init__(self):
        self.coefficiets = None
        self.intercept = None
        
    def fit(self, X, Y):
        X = np.c_[np.ones((X.shape[0], 1)), X]
        try:
            self.coefficients = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(Y)
        except np.linalg.LinAlgError:
            print("Error: Singular Matrix. consider uusing regularization or removing correlated features")
        
        self.intercept = self.coefficients[0]
        self.coefficients = self.coefficients[1:]
        
    def predict(self, X):
        return np.dot(X, self.coefficients) + self.intercept


def load_model(model_path):
    
    with open(model_path, 'rb') as f:
        return pickle.load(f)

def load_data(data_path):

    return pd.read_csv(data_path)

def preprocess_data(df):
    df = df.dropna()
    df = group_vehicle_classes(df)
    df = encode_vehicle_class(df)
    return df

def prepare_data(df, features, target_column):
    '''
    Splits the preprocessed dataset into the features and target variable for the regression

    Parameters
    ----------
    df : pd.DataFrame
        preprocessed data as a dataframe.
    features : List
        List of all the features for the regression model.
    target_column : str
        Column of the target variable.

    Returns
    -------
    X : pd.DataFrame
        Matrix of independant variables.
    Y : pd.Series
        Vector of target variable.

    '''
    X = df[features]
    Y = df[target_column]
    return X,Y

def calculate_metrics(y_true, y_pred):
    """
    Calculate evaluation metrics: MSE, RMSE, and R-squared.
    """
    mse = np.mean((y_true - y_pred)**2)
    rmse = np.sqrt(mse)
    ss_total = np.sum((y_true - np.mean(y_true))**2)
    ss_residual = np.sum((y_true - y_pred)**2)
    r2 = 1 - (ss_residual / ss_total)
    return mse, rmse, r2
        
def calculate_mse(y_true, y_pred):
    '''
    Calculates the mean squared error of the regression model

    Parameters
    ----------
    y_true : Array
        Array of true Y values from data.
    y_pred : Array
        Array of predicted Y values from the regression model.

    Returns
    -------
    float64
        Mean Squared Error of the model.

    '''
    return np.mean((y_true - y_pred)**2)

def calculate_rmse(y_true, y_pred):
    '''
    Calculates Root mean squared error of the regression model

    Parameters
    ----------
    y_true : Array
        Array of true Y values from data.
    y_pred : Array
        Array of predicted Y values from the regression model.

    Returns
    -------
    float64
        Root mean squared error of the model.

    '''
    return np.sqrt(calculate_mse(y_true, y_pred))

def calculate_r2(y_true, y_pred):
    '''
    Calculates the R squared score of the regression model

    Parameters
    ----------
    y_true : Array
        Array of true Y values from data.
    y_pred : Array
        Array of predicted Y values from the regression model. .

    Returns
    -------
    float64
        R2 score of the model.

    '''
    ss_total = np.sum((y_true - np.mean(y_true))**2)
    ss_residual = np.sum((y_true - y_pred)**2)
    return 1-(ss_residual/ss_total)

def save_metrics(mse, rmse, r2, path):
    '''
    Saves the training metrics of the model

    Parameters
    ----------
    mse : float64
        Mean squared error.
    rmse : float64
        Root mean squared error.
    r2 : float64
        r2 score.
    path : str
        Path to save the model metrics.

    Returns
    -------
    None.

    '''
    with open(path, 'w') as f:
        f.write(f"Mean Squared Error (MSE): {mse:.4f}\n")
        f.write(f"Root mean squared error (RMSE): {rmse:.4f}\n")
        f.write(f"R-squared Score: {r2:.4f}\n")

def save_predictions(predictions, output_path):
    """
    Save the predictions to a CSV file.
    """
    pd.DataFrame(predictions).to_csv(output_path, header=False, index=False)

def main():
    parser = argparse.ArgumentParser(description="Predict using trained regression model")
    parser.add_argument("--model_path", required=True, help="Path to the saved model file")
    parser.add_argument("--data_path", required=True, help="Path to the data CSV file")
    parser.add_argument("--metrics_output_path", required=True, help="Path to save evaluation metrics")
    parser.add_argument("--predictions_output_path", required=True, help="Path to save predictions")
    args = parser.parse_args()

    try:
        # Load the model
        model = load_model(args.model_path)

        # Load and preprocess the data
        df = load_data(args.data_path)
        df = preprocess_data(df)

        # Prepare features and true labels
        features = ['COEMISSIONS ', 'VC_MID-SIZE', 'VC_MINIVAN',
               'VC_PICKUP_TRUCK', 'VC_SMALL_CAR', 'VC_STATION_WAGON', 'VC_SUV',
               'VC_VAN', 'VC_VERY_SMALL_CAR']
        
        target = 'FUEL CONSUMPTION'
        
        X, Y = prepare_data(df, features, target)
        
        # Make predictions
        y_pred = model.predict(X)
        
        # Calculate and print model performance metrics
        mse = calculate_mse(Y.values, y_pred)
        rmse = calculate_rmse(Y.values, y_pred)
        r2 = calculate_r2(Y.values, y_pred)
        
        
        
        # Save metrics and predictions
        save_metrics(mse, rmse, r2, args.metrics_output_path)
        save_predictions(y_pred, args.predictions_output_path)

        print("Prediction completed successfully.")
        print(f"Metrics saved to: {args.metrics_output_path}")
        print(f"Predictions saved to: {args.predictions_output_path}")

    except FileNotFoundError as e:
        print(f"Error: File not found. {str(e)}")
    except pd.errors.EmptyDataError:
        print("Error: The data file is empty.")
    except KeyError as e:
        print(f"Error: Missing expected column in the data. {str(e)}")
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")

if __name__ == "__main__":
    main()