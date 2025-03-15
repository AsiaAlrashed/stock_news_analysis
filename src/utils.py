import pandas as pd


class DataUtils:
    @staticmethod
    def save_to_csv(data, file_path):
        """
        save data in file CSV.
        """
        df = pd.DataFrame(data)
        df.to_csv(file_path, index=False)
        print(f"Data saved in {file_path}")

    @staticmethod
    def load_from_csv(file_path):
        """
        load data from CSV.
        """
        return pd.read_csv(file_path)
