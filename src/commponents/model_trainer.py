import os 
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
# from sklearn.ensemble import (
#     AdaBoostRegressor,
#     GradientBoostingRegressor,
#     RandomForestRegressor
# )
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor,RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
#from catboost import CatBoostRegressor

#from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
#from sklearn.neighbors import KNeighborsRegressor
#from sklearn.tree import DecisionTreeRegressor
#from xgboost import XGBRegressor

from src.exception import CustomException 
from src.logger import logging
from src.utlis import save_object
from src.utlis import evaluate_models


@dataclass 
class ModelTrainerConfig:
    trained_model_path=os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()
    
    def initiate_model_trainer(self,train_arry,test_arry):
        try:
            logging.info('Split train and test input data')
            x_train,y_train,x_test,y_test=(
                train_arry[:,:-1],
                train_arry[:,-1],
                test_arry[:,:-1],
                test_arry[:,-1]
            )

            # models={
            #     "Random Forest ":RandomForestRegressor(),
            #     "Decision Tree ":DecisionTreeRegressor(),
            #     "Gradient Boosting ":GradientBoostingRegressor(),
            #     "Linear Regression ":LinearRegression(),
            #     "K-Neighbors Classifier ":KNeighborsRegressor(),
            #     "XGBClassifier ":XGBRegressor(),
            #     "CatBoosting Classifier ":CatBoostRegressor(),
            #     "Adaboost Classifier ":AdaBoostRegressor()
            # }
            # models = {

            #     "RandomForest": RandomForestRegressor(),
            #     "DecisionTree": DecisionTreeRegressor(),
            #     "GradientBoosting": GradientBoostingRegressor(),
            #     "LinearRegression": LinearRegression(),
            #     "KNeighbors": KNeighborsRegressor(),
            #     "XGBoost": XGBRegressor(),
            #     "CatBoost": CatBoostRegressor(),
            #     "AdaBoost": AdaBoostRegressor()
            # }

            models = {
                
                "LinearRegression": LinearRegression(),
                "DecisionTree": DecisionTreeRegressor(),
                "RandomForest": RandomForestRegressor(),
                "GradientBoosting": GradientBoostingRegressor(),
                "AdaBoost": AdaBoostRegressor(),
                "KNeighbors": KNeighborsRegressor(),
                "XGBoost": XGBRegressor(),
                "CatBoost": CatBoostRegressor(verbose=0)
            }



            params={
                "LinearRegression":{},				
				"DecisionTree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "RandomForest":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "GradientBoosting":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
				"AdaBoost":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                },
				"KNeighbors": {
                    'n_neighbors': [3, 5, 7, 9],
                    'weights': ['uniform', 'distance'],
                    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
                    'leaf_size': [20, 30, 40],
                    'p': [1, 2]  # 1 for Manhattan, 2 for Euclidean
                },                
                "XGBoost":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "CatBoost":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                }
                
                               
            }





            




            Model_report:dict=evaluate_models(
                x_train=x_train,
                y_train=y_train,
                x_test=x_test,
                y_test=y_test,
                models=models,
                params=params
                )
            
            


            ##To get best model score from dict
            best_model_score=max(sorted(Model_report.values()))

            ## To get Best Model Name from dict

            # best_model_name=max(sorted(Model_report.keys()))[
            #     list( Model_report.values()).index(best_model_score)
            # ]
            best_model_name = max(Model_report, key=Model_report.get) 
            #print(Model_report)
            #best_model_score = model_report[best_model_name]
            best_model=models[best_model_name]
           

            if best_model_score < 0.6:
                raise CustomException('No Best Model Found')
            
            logging.info(f'Best found model on both training and testing dataset')

            #preprocessing_obj=

            save_object(
                file_path=self.model_trainer_config.trained_model_path,
                obj=best_model
            )

            predicted=best_model.predict(x_test)

            r2_square=r2_score(y_test,predicted)
            print('The Best model is : ',best_model_name)
            print('The Best model r2_square : ',r2_square)
            return r2_square


        except Exception as e:
            raise CustomException(e,sys)
