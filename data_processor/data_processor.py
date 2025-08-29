import os
import pickle
from typing import List
import pandas as pd

class DataProcessor:
    def __init__(self, data_dir: str, pickle_dir: str):
        self.data_dir_ = data_dir
        self.pickle_dir_ = pickle_dir
        self.pickle_=[]
        self.experiment_id = ""
        self.verbose_ = True
        self.dataframes_ = []
    
    def check_existing_df(self,experiment_id,default_load=False):
        if default_load==True:
            pickle_path = os.path.join(self.pickle_dir_, "RipsComp.pkl")
            self.pickle_=pd.read_pickle(pickle_path)
            print("RipsComp.pkl has been loaded")
        pickle_path = os.path.join(self.pickle_dir_, experiment_id+"_joined.pkl")
        if os.path.exists(pickle_path):
            if self.verbose:
                print(f"Loading joined dataframe from pickle: {pickle_path}")
                with open(pickle_path, "rb") as f:
                    self.pickle_ = pickle.load(f)
        else:
            print("No experiment found")
                    
    def load_csvs(self,experiment_id,default_load=False) -> List[pd.DataFrame]:
        """Load all CSV files from the data directory."""
        self.check_existing_df(experiment_id,default_load=default_load)
        if self.pickle_==[]:
            print(f"Loading CSVs from {self.data_dir_}")
            for file in os.listdir(self.data_dir_):
                if file.endswith(".csv"):
                    df = pd.read_csv(os.path.join(self.data_dir_, file))
                    self.dataframes_.append(df)
                    print("Loading csv: " + str(file))
            return self.dataframes_

    def join_dataframes(self) -> pd.DataFrame:
        print("Joining dataframes...")
        master_df=self.dataframes_[0]
        for df in self.dataframes_[1:]:
            master_df = pd.merge(master_df,df, on="Time",how="outer")
            if "StepCount" in master_df.columns:
                master_df=master_df.drop(columns=["StepCount"])
            master_df=self.trim_dataframe(master_df) 
        print("Dataframe columns: " + df.columns)
        return master_df

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean the joined dataframe."""
        if self.verbose_:
            print("Cleaning data...")
        # Drop time columns
        df = df.drop(columns=[col for col in df.columns if "time" in col.lower()], errors='ignore')
        # Drop columns with only NaNs
        df = df.dropna(axis=1, how='all')
        # Fill remaining NaNs with forward fill or zero
        #df = df.fillna(method='ffill').fillna(0)
        df=df.ffill()
        # Normalize each column to itself
        df = df.apply(lambda x: (x - x.mean()) / x.std() if x.std() != 0 else x)

        return df
    
    def trim_dataframe(self,df):
        # Identify rows where all columns are non-null
        valid_rows = df.notnull().all(axis=1)

        # Find the last index where all columns are valid
        if valid_rows.any():
            last_valid_index = valid_rows[valid_rows].index[-1]
            trimmed_df = df.loc[:last_valid_index]
        else:
            trimmed_df = pd.DataFrame(columns=df.columns)  # Return empty if no valid row

        return trimmed_df
