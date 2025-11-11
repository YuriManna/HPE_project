from Dataset import Dataset
from model import *

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
    walmart.drop_columns(["Date"])

    walmart.convert_nominal(["Type"])

    walmart.to_categorical(["Store", "Dept"])
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

    walmart.to_categorical(["Store", "Dept"])
    walmart.export_data("../dataset_cleaned_with_MarkDown.csv")


def models():
    dataset = Dataset("../dataset_cleaned_with_MarkDown.csv")
    dataset.drop_columns(columns=["Fuel_Price", "CPI", "Unemployment"])
    #dataset.drop(columns=["Day", "Month", "Year"])

    lin_reg_model = LinReg(dataset)
    lin_reg_model.split_data(target_column="Weekly_Sales")
    lin_reg_model.scale_data()
    '''
    lin_reg_model.train()
    lin_reg_model.evaluate()
    print(lin_reg_model.model.coef_)

    lin_reg_model = RidgeModel(dataset)
    lin_reg_model.split_data(target_column="Weekly_Sales")
    lin_reg_model.train()
    lin_reg_model.evaluate()

    lin_reg_model = LassoModel(dataset)
    lin_reg_model.split_data(target_column="Weekly_Sales")
    lin_reg_model.train()
    lin_reg_model.evaluate()

    lin_reg_model = ElasticNetModel(dataset)
    lin_reg_model.split_data(target_column="Weekly_Sales")
    lin_reg_model.train()
    lin_reg_model.evaluate()

    # Non-linear models
    dataset.data['Store'] = dataset.data['Store'].astype('category')
    dataset.data['Dept'] = dataset.data['Dept'].astype('category')

    lin_reg_model = DecisionTreeRegressorModel(dataset)
    lin_reg_model.split_data(target_column="Weekly_Sales")
    lin_reg_model.train()
    lin_reg_model.evaluate()

    lin_reg_model = RandomForestRegressorModel(dataset)
    lin_reg_model.split_data(target_column="Weekly_Sales")
    lin_reg_model.train()
    lin_reg_model.evaluate()

    lin_reg_model = XGBoostRegressorModel(dataset)
    lin_reg_model.split_data(target_column="Weekly_Sales")
    lin_reg_model.train()
    lin_reg_model.evaluate()

    # Neural Networks
    ffn_model = FeedforwardNN(dataset)
    ffn_model.split_data(target_column="Weekly_Sales")
    ffn_model.train(epochs=50)
    ffn_model.evaluate()
    '''
'''
    lstm_model = LSTM(dataset, timesteps=5)
    lstm_model.split_data(target_column="Weekly_Sales")

    X_seq, y_seq = model.prepare_sequences(target_column="Weekly_Sales")
    lstm_model.scale_data()

    # Step 4: build and train
    model.build_model()
    model.train(epochs=50, batch_size=32)

    # Step 5: evaluate
    model.evaluate()
    '''

models()
#clean_dataset_with_MarkDown()
#clean_dataset_without_MarkDown()

#walmart.standardize_dataset(['Store', 'Dept', 'Weekly_Sales'])