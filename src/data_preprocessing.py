# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 00:04:40 2024

@author: samvi
"""

import pandas as pd
import numpy as np
import pickle

def load_data(path):
    '''
    Loads data from a CSV file

    Parameters
    ----------
    path : str
        file path of the dataset.

    Returns
    -------
    pd.DataFrame
    
    '''
    return pd.read_csv(path)

def drop_missing_observations(df):
    '''
    Drops missing observations(NaN values) from the dataset to ensure no missing value errors are thrown 

    Parameters
    ----------
    df : pd.DataFrame
        Dataset that has been imported as a dataframe.

    Returns
    -------
    pd.DataFrame
        Dataframe with observations with missing values dropped.

    '''
    return df.dropna()

def simplify_vehicle_classes(df):
    '''
    Simplifies the groupings of the vehicle classes so that fewer variables are necessary in the model

    Parameters
    ----------
    df : pd.DataFrame
        Imported dataset with missing values removed.

    Returns
    -------
    df : pd.DataFrame
        dataset with the new combined vehicle classes based on the mapping. 

    '''
    vehicle_class_mapping = {
        'COMPACT': 'SMALL_CAR',
        'SUBCOMPACT': 'SMALL_CAR',
        'VAN - PASSENGER': 'VAN',
        'VAN - CARGO': 'VAN',
        'STATION WAGON - SMALL': 'STATION_WAGON',
        'STATION WAGON - MID-SIZE': 'STATION_WAGON',
        'SUV': 'SUV',
        'PICKUP TRUCK - STANDARD': 'PICKUP_TRUCK',
        'PICKUP TRUCK - SMALL': 'PICKUP_TRUCK',
        'MINIVAN': 'MINIVAN',
        'MINICOMPACT': 'VERY_SMALL_CAR',
        'TWO-SEATER': 'VERY_SMALL_CAR',
        'FULL-SIZE': 'FULL-SIZE',
        'MID-SIZE': 'MID-SIZE'
    }
    df['VEHICLE CLASS'] = df['VEHICLE CLASS'].map(vehicle_class_mapping)
    return df

def encode_vehicle_class(df):
    '''
    Creates dummy variables for each of the combined vehicle classes. The first class is dropped to prevent multicolinearity issues

    Parameters
    ----------
    df : pd.DataFrame
        Dataset

    Returns
    -------
    pd.DataFrame
        Dataset with new dummy varaiable columns for vehicle class.

    '''
    df_encoded = pd.get_dummies(df, columns = ['VEHICLE CLASS'], prefix = 'VC', drop_first=True)
    
    # Convert boolean dummy variables to binary (0 and 1)
    dummy_columns = [col for col in df_encoded.columns if col.startswith('VC_')]
    df_encoded[dummy_columns] = df_encoded[dummy_columns].astype(int)
    return df_encoded

class MinMaxScaler:
    def __init__(self):
        self.min_ = None
        self.scale_ = None
    
    def fit(self, X):
        self.min_ = X.min()
        self.scale_ = X.max() - self.min_
        
    def transform(self, X):
        return (X-self.min_)/self.scale_
    
    def inverse_transform(self, X):
        return self.scale_*X+self.min_
      
def scale_numerical_features(df, columns_to_scale):
    '''
    Scales the numerical features using minmaxscaler (inbuilt)

    Parameters
    ----------
    df : pd.DataFrame
        Dataset.
    columns_to_scale : List
        List of the columns that need to be scaled

    Returns
    -------
    df : pd.DataFrame
        Dataset with the scaling applied.
    scalers : dict
        Saved parameters for scaling (min and max) for each column so that data can be inverse scaled.

    '''
    scalers = {}
    for column in columns_to_scale:
        scaler = MinMaxScaler()
        scaler.fit(df[[column]])
        df[column] = scaler.transform(df[[column]])
        scalers[column] = scaler
    return df, scalers

def preprocess_data(input_file, output_file, scaler_file, apply_scaling=False):
    '''
    Loads, removes missing values, simplifies the vehicle classs and creates dummy variables for the original dataset before exporting it as a new csv file

    Parameters
    ----------
    input_file : str
        File path for the raw dataset.
    output_file : str
        File path for the preprocessed data to be stored.
    scaler_file : str
        File path for the saved scaler data to be stored.
    apply_scaling : Boolean, optional
        Determines if the function applies scaling or not. The default is False.

    Returns
    -------
    None.

    '''
    df = load_data(input_file)
    df = drop_missing_observations(df)
    
    # Groups similar vehicle classes - based on different vehicle classes that show similar patterns in fuel consumption
    df = simplify_vehicle_classes(df)
    
    # Create dummy variables for the vehicle classes
    df = encode_vehicle_class(df)
    
    # Scales numerical variables to the range [0,1] using minmaxscaler
    # Minmax scaler is chosen as the numerical variables have different ranges
    scalers = None
    if apply_scaling:
        columns_to_scale = ['ENGINE SIZE', 'CYLINDERS', 'FUEL CONSUMPTION', 'COEMISSIONS ']
        df, scalers = scale_numerical_features(df, columns_to_scale)
    
    df.to_csv(output_file, index = False)
    
    # Saves the scaler parameters if scaling is used as a pkl file
    if scalers:
        with open(scaler_file, 'wb') as f:
            pickle.dump(scalers, f)

if __name__ == "__main__":
    input_file = r"C:\Users\samvi\OneDrive\Desktop\Semester\IML\Sameer_Rangwala_A1\regression_task\data\fuel_train.csv"
    output_file = r"C:\Users\samvi\OneDrive\Desktop\Semester\IML\Sameer_Rangwala_A1\regression_task\data\preprocessed_fuel_train.csv"
    scaler_file = r"C:\Users\samvi\OneDrive\Desktop\Semester\IML\Sameer_Rangwala_A1\regression_task\data\scalers.pkl"
            
    apply_scaling = False
    
    preprocess_data(input_file, output_file, scaler_file, apply_scaling)
    print("Data Preprocessing completed. Preprocessed data saved.")
    if apply_scaling:
        print("Scalers saved")
    else:
        print("No scaling applied")
        
