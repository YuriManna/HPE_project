import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


class Dataset:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None
        self.load_data()

    def load_data(self):
        """Load data from a CSV file."""
        self.data = pd.read_csv(self.file_path) #, na_values=["", "NaN", "None"], keep_default_na=False)
        return self.data

    def export_data(self, file_name):
        """Export data in a CSV file."""
        if self.data is None:
            print("Data not loaded. Please load the data first.")
            return
        print("Loading DataFrame")
        self.data.to_csv(file_name, index = False)

    def visualize_dataset(self, type=None, n_cols=5):
        """View the first few rows of the dataset."""
        if self.data is None:
            print("Data not loaded. Please load the data first.")
            return
        if type is None:
            type = ["head", "info", "columns"]
        for t in type:
            if t == "head":
                print(self.data.head(n=n_cols))
            if t == "info":
                print(self.data.describe(include="all"))
            if t == "columns":
                print(self.data.columns)

    def left_join_dataframe(self, new_data, keys):
        """Join a Dataframe from a given key"""
        if self.data is None:
            print("Data not loaded. Please load the data first.")
            return
        print("-------- Joining the datasets ---------------")
        self.data = pd.merge(self.data, new_data, on=keys, how="left")

    def check_equal_columns(self, col1, col2):
        if self.data is None:
            print("Data not loaded. Please load the data first.")
            return False
        return self.data[col1].equals(self.data[col2])

    def drop_columns(self, columns):
        """Drop specified columns from the dataset."""
        if self.data is None:
            print("Data not loaded. Please load the data first.")
            return
        print(f"---------------------- Dropping columns {columns} --------------------------")
        self.data = self.data.drop(columns=columns)

    def drop_negative_values(self, columns):
        if self.data is None:
            print("Data not loaded. Please load the data first.")
            return
        for col in columns:
            print(f"---------------------- Dropping rows by negative values in {col} --------------------------")
            self.data = self.data[self.data[col] >= 0]

    def set_zeros_MarkDown(self, columns):
        if self.data is None:
            print("Data not loaded. Please load the data first.")
            return
        mask = self.data[columns].notna().any(axis=1)
        self.data.loc[mask, columns] = self.data.loc[mask, columns].fillna(0)

    def drop_nan_rows(self, columns):
        if self.data is None:
            print("Data not loaded. Please load the data first.")
            return
        for col in columns:
            print(f"---------------------- Dropping rows by NaN values in {col} --------------------------")
            self.data = self.data.dropna(subset=[col])

    def split_date(self, date_col):
        if self.data is None:
            print("Data not loaded. Please load the data first.")
            return
        print(f"---------------------- Splitting date in column {date_col} --------------------------")
        self.data[date_col] = pd.to_datetime(self.data[date_col], format='%Y-%m-%d', errors='coerce')

        day = self.data[date_col].dt.day
        month = self.data[date_col].dt.month
        year = self.data[date_col].dt.year

        self.data['Years_since_start'] = year - year.min()

        self.data['Month_sin'] = np.sin(2 * np.pi * month / 12)
        self.data['Month_cos'] = np.cos(2 * np.pi * month / 12)
        self.data['Day_sin'] = np.sin(2 * np.pi * day / 31)
        self.data['Day_cos'] = np.cos(2 * np.pi * day / 31)

    def convert_nominal(self, columns):
        """Substitute nominal columns with dummy variables"""
        if self.data is None:
            print("Data not loaded. Please load the data first.")
            return
        for col in columns:
            dummies = pd.get_dummies(self.data[col], prefix=col, dummy_na=False)
            self.data = pd.concat([self.data, dummies], axis=1)
            self.data = self.data.drop(columns=[col])

    def standardize_dataset(self, exclude):
        if self.data is None:
            print("Data not loaded. Please load the data first.")
            return
        print(f"---------------------- Start standardization --------------------------")
        df_scaled = self.data.copy()
        numeric_cols = df_scaled.select_dtypes(include=['float64', 'int64']).columns.tolist()
        numeric_cols = [c for c in numeric_cols if c not in exclude]
        print("Numerical columns to be standardized:", numeric_cols)

        scaler = StandardScaler()
        df_scaled[numeric_cols] = scaler.fit_transform(df_scaled[numeric_cols])
        print(f"---------------------- Standardization complete --------------------------")

        return df_scaled, scaler


    def to_categorical(self, columns):
        if self.data is None:
            print("Data not loaded. Please load the data first.")
            return
        for col in columns:
            self.data[col] = self.data[col].astype('category')