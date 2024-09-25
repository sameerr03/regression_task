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
    return pd.read_csv(path)

def prepare_data(df, features, target_column):
    X = df[features]
    Y = df[target_column]
    return X,Y

def calculate_mse(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)

def calculate_rmse(y_true, y_pred):
    return np.sqrt(calculate_mse(y_true, y_pred))

def calculate_r2(y_true, y_pred):
    ss_total = np.sum((y_true - np.mean(y_true))**2)
    ss_residual = np.sum((y_true - y_pred)**2)
    return 1-(ss_residual/ss_total)

def save_model(model, path):
    with open(path, 'wb') as f:
        pickle.dump(model, f)

def save_metrics(mse, rmse, r2, path):
    with open(path, 'w') as f:
        f.write(f"Mean Squared Error (MSE): {mse:.4f}\n")
        f.write(f"Root mean squared error (RMSE): {rmse:.4f}\n")
        f.write(f"R-squared Score: {r2:.4f}\n")

def save_predictions(y_pred, path):
    pd.DataFrame(y_pred, columns=['Prediction']).to_csv(path, index=False)

def main():
    data_path = r"C:\Users\samvi\OneDrive\Desktop\Semester\IML\Sameer_Rangwala_A1\regression_task\data\preprocessed_fuel_train.csv"
    df = load_pdata(data_path)
    
    features = ['COEMISSIONS ']
    target = 'FUEL CONSUMPTION'
    X,Y = prepare_data(df, features, target)
    
    model = LinearRegression()
    model.fit(X.values, Y.values)
    
    if model.coefficients is None:
        print("Model training failed")
        return
    
    y_pred = model.predict(X.values)
    
    mse = calculate_mse(Y.values, y_pred)
    rmse = calculate_rmse(Y.values, y_pred)
    r2 = calculate_r2(Y.values, y_pred)
    
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"Root Mean Squared Error: {rmse:.4f}")
    print(f"R-squared Score: {r2:.4f}")

    # Save the model
    model_path = r"C:\Users\samvi\OneDrive\Desktop\Semester\IML\Sameer_Rangwala_A1\regression_task\models\regression_model1.pkl"
    save_model(model, model_path)
    print(f"Model saved to {model_path}")
    
    # Save metrics
    metrics_path = r"C:\Users\samvi\OneDrive\Desktop\Semester\IML\Sameer_Rangwala_A1\regression_task/results/train_metrics.txt"
    save_metrics(mse, rmse, r2, metrics_path)
    print(f"Metrics saved to {metrics_path}")
  
    # Save predictions
    predictions_path = r"C:\Users\samvi\OneDrive\Desktop\Semester\IML\Sameer_Rangwala_A1\regression_task/results/train_predictions.csv"
    save_predictions(y_pred, predictions_path)
    print(f"Predictions saved to {predictions_path}")
    
if __name__=="__main__":
    main()