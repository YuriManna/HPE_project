import os
from Dataset import Dataset
from model import *
from tensorflow.keras import mixed_precision

def create_dataset():
    walmart = Dataset("../train.csv")
    walmart.visualize_dataset(["columns"])

    stores = Dataset("../stores.csv")
    stores.visualize_dataset(["columns"])

    info = Dataset("../features.csv")
    info.visualize_dataset(["columns"])

    info.left_join_dataframe(stores.data, ["Store"])
    info.visualize_dataset(["columns"])
    info.export_data("../temp.csv")

    walmart.left_join_dataframe(info.data, ["Store", "Date"])
    walmart.visualize_dataset(["columns"])

    if walmart.check_equal_columns("IsHoliday_x", "IsHoliday_y"):
        walmart.drop_columns("IsHoliday_y")
        walmart.data.rename(columns={'IsHoliday_x': 'IsHoliday'}, inplace=True)

    walmart.visualize_dataset(["columns"])

    walmart.export_data("../dataset.csv")

def clean_dataset_without_MarkDown():
    walmart = Dataset("../dataset.csv")
    walmart.visualize_dataset(["info"])

    walmart.drop_negative_values(["Weekly_Sales"])
    walmart.visualize_dataset(["info"])
    print(walmart.data.isna().sum())
    print("---------------------------")

    walmart.drop_columns(["MarkDown1", "MarkDown2", "MarkDown3", "MarkDown4", "MarkDown5"])

    print(walmart.data.isna().sum())

    walmart.split_date("Date")
    #walmart.drop_columns(["Date"])

    walmart.convert_nominal(["Type"])

    #walmart.to_categorical(["Store", "Dept"])
    walmart.export_data("../dataset_cleaned_without_MarkDown.csv")


def clean_dataset_with_MarkDown():
    walmart = Dataset("../dataset.csv")
    walmart.visualize_dataset(["info"])

    walmart.drop_negative_values(["Weekly_Sales"])
    walmart.visualize_dataset(["info"])
    print(walmart.data.isna().sum())
    print("---------------------------")

    mark_down_columns = ["MarkDown1", "MarkDown2", "MarkDown3", "MarkDown4", "MarkDown5"]
    walmart.set_zeros_MarkDown(mark_down_columns)
    print(walmart.data.isna().sum())
    print("---------------------------")

    walmart.drop_nan_rows(mark_down_columns)
    print(walmart.data.isna().sum())
    walmart.visualize_dataset(["info"])
    print("---------------------------")

    walmart.drop_negative_values(mark_down_columns)
    print(walmart.data.isna().sum())
    walmart.visualize_dataset(["info"])
    print("---------------------------")

    walmart.split_date("Date")

    walmart.drop_columns(["Date"])

    walmart.convert_nominal(["Type"])

    #walmart.to_categorical(["Store", "Dept"])
    walmart.export_data("../dataset_cleaned_with_MarkDown.csv")


def models():
    dataset = Dataset("../dataset_cleaned_without_MarkDown.csv")
    #dataset =  Dataset("/mnt/c/Users/yurim/Documents/Work/Formazione_Experis/codice/Walmart/dataset_cleaned_without_MarkDown.csv")
    dataset.drop_columns(columns=["Fuel_Price", "CPI", "Unemployment"])
    #dataset.drop(columns=["Day", "Month", "Year"])

    ffn_model = FeedforwardNN(dataset)
    ffn_model.split_data(target_column="Weekly_Sales")
    ffn_model.train(epochs=1000, batch_size=1024, patience=100)
    ffn_model.evaluate()
    
    lin_reg_model = LinReg(dataset)
    lin_reg_model.split_data(target_column="Weekly_Sales")
    lin_reg_model.scale_data()
    lin_reg_model.train()
    lin_reg_model.evaluate()
    print(lin_reg_model.model.coef_)
    
    lin_reg_model = RidgeModel(dataset)
    lin_reg_model.split_data(target_column="Weekly_Sales")
    lin_reg_model.train()
    lin_reg_model.evaluate(plot=True)

    lin_reg_model = LassoModel(dataset)
    lin_reg_model.split_data(target_column="Weekly_Sales")
    lin_reg_model.train()
    lin_reg_model.evaluate(plot=True)

    lin_reg_model = ElasticNetModel(dataset)
    lin_reg_model.split_data(target_column="Weekly_Sales")
    lin_reg_model.train()
    lin_reg_model.evaluate(plot=True)

    # Non-linear models
    dataset.data['Store'] = dataset.data['Store'].astype('category')
    dataset.data['Dept'] = dataset.data['Dept'].astype('category')

    lin_reg_model = DecisionTreeRegressorModel(dataset)
    lin_reg_model.split_data(target_column="Weekly_Sales")
    lin_reg_model.train()
    lin_reg_model.evaluate(plot=True)

    lin_reg_model = RandomForestRegressorModel(dataset)
    lin_reg_model.split_data(target_column="Weekly_Sales")
    lin_reg_model.train()
    lin_reg_model.evaluate(plot=True)

    lin_reg_model = XGBoostRegressorModel(dataset)
    lin_reg_model.split_data(target_column="Weekly_Sales")
    lin_reg_model.train()
    lin_reg_model.evaluate(plot=True)
    
    # Neural Networks

    '''
    '''
    lstm_model = LSTM(dataset, timesteps=5)
    lstm_model.split_data(target_column="Weekly_Sales")
    X, y = lstm_model.prepare_sequences()
    lstm_model.scale_data(feature_range=(-1, 1))
    lstm_model.build_model()
    with tf.device('/GPU:0'):
        lstm_model.train(epochs=50, batch_size=32, X_seq=X, y_seq=y, patience=20)
    lstm_model.evaluate()

def __main__():
    mixed_precision.set_global_policy('mixed_float16')
    print("GPUs detected:", tf.config.list_physical_devices('GPU'))
    print("TensorFlow built with CUDA:", tf.test.is_built_with_cuda())
    print("Is GPU available:", tf.test.is_gpu_available())
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # hides info logs
    os.environ['TF_XLA_FLAGS'] = '--xla_gpu_autotune_level=1'

    #clean_dataset_with_MarkDown()
    #create_dataset()
    #clean_dataset_without_MarkDown()
    models()
    #walmart.standardize_dataset(['Store', 'Dept', 'Weekly_Sales'])

if __name__ == "__main__":
    __main__()