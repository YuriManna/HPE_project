import pandas as pd
from keras.src.datasets.boston_housing import load_data


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
        else:
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
        else:
            print(f"---------------------- Dropping column {columns} --------------------------")
            self.data = self.data.drop(columns=columns)

    def fill_NaN (self):
        """Fill NaN values of the dataset"""
        if self.data is None:
            print("Data not loaded. Please load the data first.")
        else:
            for cols in self.data.columns:
                if self.data[cols].isnull().any():
                    if self.data[cols].dtype == 'int64':
                        self.fill_NaN_int(cols)
                    elif self.data[cols].dtype == 'float64':
                        self.fill_NaN_float(cols)
                    else:
                        self.fill_NaN_object(cols)

    def fill_NaN_int (self, col):
        """Fill NaN values of a column of integers"""
        median = self.data[col].median()
        self.data[col] = self.data[col].fillna(median)

    def fill_NaN_float (self, col):
        """Fill NaN values of a column of floats"""
        mean = self.data[col].mean()
        self.data[col] = self.data[col].fillna(mean)

    def fill_NaN_object (self, col):
        """Fill NaN values of a column of objects"""
        mode = self.data[col].mode()[0]
        self.data[col] = self.data[col].fillna(mode)

    def convert_string_to_number (self, col):
        """Convert column of type object containing numbers to type int or float"""
        self.data[col] = pd.to_numeric(self.data[col], errors='coerce')

    def map_column (self, col, mapping):
        """Mappa i valori di una colonna secondo il dizionario dato."""
        self.data[col] = self.data[col].map(mapping)

    def convert_nominal (self, columns):
        """Substitute nominal columns with dummy variables"""
        for col in columns:
            dummies = pd.get_dummies(self.data[col], prefix=col, dummy_na=False)
            self.data = pd.concat([self.data, dummies], axis=1)
            self.data = self.data.drop(columns=[col])