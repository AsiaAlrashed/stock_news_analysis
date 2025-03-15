import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from src.text_preprocessing import TextPreprocessor
from src.data_augmentation import DataAugmentor


class DataPreparer:
    "A class for processing and preparing data for training."

    def __init__(self):
        self.text_preprocessor = TextPreprocessor()
        self.data_augmentor = DataAugmentor()
        self.label_encoder = LabelEncoder()
        self.source_encoder = LabelEncoder()
        self.scaler = MinMaxScaler()

    def prepare_data(self, df):
        """
        Data cleaning, augmentation, and encoding for training.
        """
        df["clean_headline"] = df["headline"].apply(self.text_preprocessor.clean_text)
        df["augmented_headline"] = df["clean_headline"].apply(
            self.data_augmentor.augment_text
        )

        df["final_label_encoded"] = self.label_encoder.fit_transform(df["final_label"])
        df["source_encoded"] = self.source_encoder.fit_transform(df["source"])
        df["source_scaled"] = self.scaler.fit_transform(
            df["source_encoded"].values.reshape(-1, 1)
        )

        return df
