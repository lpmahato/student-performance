import sys
import os
import pandas as pd

from src.utils import load_object
from src.exception import CustomException


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path=os.path.join("artifacts","model.pkl")
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            print("Before Loading")
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            print("After Loading")
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds
        except Exception as e:
            raise CustomException(e, sys)



class CustomPipeline:
    def __init__(self, gender: str, race_ethnicity: str, parent_level_of_education: str, lunch: str, test_preperation_course: str, reading_score:int, writing_score: int):
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parent_level_of_education = parent_level_of_education
        self.lunch = lunch
        self.test_preperation_course = test_preperation_course
        self.reading_score = reading_score
        self.writing_score = writing_score


    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                'gender': [self.gender],
                'race_ethnicity': [self.race_ethnicity],
                'parent_level_of_education': [self.parent_level_of_education],
                'lunch': [self.lunch],
                'test_preperation_course': [self.test_preperation_course],
                'reading_score': [self.reading_score],
                'writing_score': [self.writing_score],
            }
            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            raise CustomException(e, sys)