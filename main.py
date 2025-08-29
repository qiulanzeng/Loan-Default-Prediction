from src.components.data_preprocessing import DataPreprocessing
from src.components.model_trainer import ModelTrainer
from src.components.model_validation import ModelValidator
import pandas as pd
from sklearn.model_selection import train_test_split
import os
from src.logging import logger

STAGE_NAME = "Load dataset stage"
try:
     logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
     df = pd.read_csv("data/Loan_Default_kaggle.csv")
     target = 'Status'
     X = df.drop(columns=target)  # Replace with actual target column
     y = df[target]

     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

     if not os.path.exists('X_test.csv'):
        X_test.to_csv('data/X_test.csv', index=False)
        y_test.to_csv('data/y_test.csv', index=False)

except Exception as e:
     logger.exception(e)
     raise e

STAGE_NAME = "Data Preprocessing"
try:
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
   preprocessor_builder = DataPreprocessing(config={})
   preprocessor, X_train_preprocessed, y_train_preprocessed = preprocessor_builder.build_preprocessor(X_train, y_train)

   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        logger.exception(e)
        raise e



STAGE_NAME = "Model Trainer stage"
try: 
   logger.info(f"*******************")
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
   trainer = ModelTrainer(config={})
   trained_model = trainer.train(preprocessor, X_train_preprocessed, y_train_preprocessed)
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        logger.exception(e)
        raise e




STAGE_NAME = "Model Evaluation stage"
try: 
   logger.info(f"*******************")
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
   validator = ModelValidator()
   validator.evaluate(trained_model, X_test, y_test)
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        logger.exception(e)
        raise e
