from Dataset import Dataset

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
