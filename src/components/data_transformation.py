# header file
import sys, os
from os.path import dirname, join, abspath
sys.path.insert(0, abspath(join(dirname(__file__), '../..')))

# import libraries
import os, sys
from src.logger import logging
from src.exception import CustomException
from src.utils import save_obj, load_obj
from dataclasses import dataclass

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer  # Handling Missing Values
from sklearn.preprocessing import StandardScaler  # Handling Feature Scaling
from sklearn.preprocessing import OneHotEncoder  # OneHot Encoding
# pipelines
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


# create a Data TransformationConfig Class
@dataclass
class DataTransformationConfig:
    preprocessor_file_path = os.path.join(os.getcwd(), "artifacts", "preprocessor.pkl")
    clean_train_file_path = os.path.join(os.getcwd(), "artifacts", "clean_train.csv")
    clean_test_file_path = os.path.join(os.getcwd(), "artifacts", "clean_test.csv")


# create Data Transformation class
class DataTransformation:

    def __init__(self):
        self.transformation_config = DataTransformationConfig()

    def getPreprocessorObject(self):
        # separate columns based on data types
        categorical_cols = ['season', 'weathersit', 'month', 'day_of_week', 'hour']
        numerical_cols = ['temp', 'atemp', 'hum', 'windspeed']

        # Numerical Pipeline
        num_pipeline = Pipeline(
            steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ]
        )

        # Categorical Pipeline
        cat_pipeline = Pipeline(
            steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ]
        )

        preprocessor = ColumnTransformer([
            ('num_pipeline', num_pipeline, numerical_cols),
            ('cat_pipeline', cat_pipeline, categorical_cols)
        ])
        logging.info('Preprocessor created successfully!')

        return preprocessor

    def initiateDataTransformation(self, train_path, test_path):
        logging.info('Data Transformation has started')
        try:
            # read test and train data
            train_df = pd.read_csv(train_path)
            logging.info('Train data read successfully')

            test_df = pd.read_csv(test_path)
            logging.info('Test data read successfully')

            # convert date column to datetime
            train_df['dteday'] = pd.to_datetime(train_df['dteday'])
            test_df['dteday'] = pd.to_datetime(test_df['dteday'])

            # extract new features from date column
            train_df['month'] = train_df['dteday'].dt.month
            train_df['day_of_week'] = train_df['dteday'].dt.dayofweek
            train_df['hour'] = train_df['dteday'].dt.hour

            test_df['month'] = test_df['dteday'].dt.month
            test_df['day_of_week'] = test_df['dteday'].dt.dayofweek
            test_df['hour'] = test_df['dteday'].dt.hour

            # drop the original date column as we have extracted useful features
            train_df = train_df.drop('dteday', axis=1)
            test_df = test_df.drop('dteday', axis=1)

            # split dependent and independent features
            X_train, y_train = train_df.drop(['instant', 'cnt'], axis=1), train_df['cnt']
            X_test, y_test = test_df.drop(['instant', 'cnt'], axis=1), test_df['cnt']
            logging.info('Splitting of Dependent and Independent features is successful')

            #get preprocess and pre-processor the content
            preprocessor=self.getPreprocessorObject()
            X_train_arr=preprocessor.fit_transform(X_train)
            logging.info('X_train successfully pre-processed')

            X_test_arr=preprocessor.transform(X_test)
            logging.info('X_test successfully pre-processed')

            # combine X_train_arr with y_train and vice versa
            clean_train_arr = np.c_[X_train_arr, np.array(y_train)]
            clean_test_arr = np.c_[X_test_arr, np.array(y_test)]
            logging.info('Concatenation of cleaned arrays is successful')

            save_obj(self.transformation_config.preprocessor_file_path,preprocessor)
            logging.info('Pre-processor successfully saved')

            return (
                clean_train_arr, clean_test_arr
            )

        except Exception as e:
            raise CustomException(e,sys)
if __name__=="__main__":    
    data_transformation=DataTransformation()
    data_transformation.initiateDataTransformation(train_path='artifacts\\train.csv',test_path='artifacts\\test.csv')


