# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 02:29:17 2024

@author: samvi
"""

import numpy as np
import pandas as pd
import pickle
import os

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

def load_pdata(path):
    '''
    Loads preprocessed data

    Parameters
    ----------
    path : str
        file path of preprocessed data.

    Returns
    -------
    pd.DataFrame
        preprocessed data in a dataframe.

    '''
    return pd.read_csv(path)

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

def save_model(model, path):
    '''
    Saves the  regression model as a pkl file

    Parameters
    ----------
    model : LinearRegression Object
        Trained linear regression model.
    path : str
        path to store the trained model.

    Returns
    -------
    None.

    '''
    with open(path, 'wb') as f:
        pickle.dump(model, f)

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

def save_predictions(y_pred, path):
    '''
    Saves the predictions of the model as a csv file

    Parameters
    ----------
    y_pred : Array
        Array of Y values predicted by the model.
    path : str
        Path to store the predicted Y values.

    Returns
    -------
    None.

    '''
    pd.DataFrame(y_pred, columns=['Prediction']).to_csv(path, index=False)

def main():
    # Load the data
    data_path = r"C:\Users\samvi\OneDrive\Desktop\Semester\IML\Sameer_Rangwala_A1\regression_task\data\preprocessed_fuel_train.csv"
    df = load_pdata(data_path)
    
    # Determine features used for regression
    features = ['COEMISSIONS ', 'VC_MID-SIZE', 'VC_MINIVAN',
           'VC_PICKUP_TRUCK', 'VC_SMALL_CAR', 'VC_STATION_WAGON', 'VC_SUV',
           'VC_VAN', 'VC_VERY_SMALL_CAR']
    target = 'FUEL CONSUMPTION'
    
    # Split the dataset
    X,Y = prepare_data(df, features, target)
    
    # Fit the model
    model = LinearRegression()
    model.fit(X.values, Y.values)
    
    # incase X.TX matrix is not invertible (No penrose inverse)
    if model.coefficients is None:
        print("Model training failed")
        return
    
    # Predict Y values using trained model
    y_pred = model.predict(X.values)
    
    # Calculate and print model performance metrics
    mse = calculate_mse(Y.values, y_pred)
    rmse = calculate_rmse(Y.values, y_pred)
    r2 = calculate_r2(Y.values, y_pred)
    
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"Root Mean Squared Error: {rmse:.4f}")
    print(f"R-squared Score: {r2:.4f}")

    # Save the model
    model_path = r"C:\Users\samvi\OneDrive\Desktop\Semester\IML\Sameer_Rangwala_A1\regression_task\models\regression_model2.pkl"
    save_model(model, model_path)
    print(f"Model saved to {model_path}")
    
    # Save metrics
    metrics_path = r"C:\Users\samvi\OneDrive\Desktop\Semester\IML\Sameer_Rangwala_A1\regression_task/results/train_metrics2.txt"
    save_metrics(mse, rmse, r2, metrics_path)
    print(f"Metrics saved to {metrics_path}")
  
    # Save predictions
    predictions_path = r"C:\Users\samvi\OneDrive\Desktop\Semester\IML\Sameer_Rangwala_A1\regression_task/results/train_predictions2.csv"
    save_predictions(y_pred, predictions_path)
    print(f"Predictions saved to {predictions_path}")
    
if __name__=="__main__":
    main()