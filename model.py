from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge as SKRidge
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import tensorflow as tf
import numpy as np
import time

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM as KerasLSTM, Dense, Input, Dropout, BatchNormalization


class RegressionModel:
    def __init__(self, dataset):
        self.dataset = dataset
        self.model = None

    def split_data(self, target_column, group_cols=['Store', 'Dept'], test_size=0.2):
        X_train_list, X_test_list, y_train_list, y_test_list = [], [], [], []
        df = self.dataset.data.copy().sort_values(group_cols + ['Date'])

        for _, group in df.groupby(group_cols, observed=True):
            y = group[target_column].values
            X = group.drop(columns=[target_column])

            split_index = int(len(group) * (1 - test_size))
            X_train_list.append(X.iloc[:split_index])
            X_test_list.append(X.iloc[split_index:])
            y_train_list.append(y[:split_index])
            y_test_list.append(y[split_index:])

        self.X_train = pd.concat(X_train_list)
        self.X_test = pd.concat(X_test_list)
        self.y_train = np.concatenate(y_train_list)
        self.y_test = np.concatenate(y_test_list)

        self.X_train = self.X_train.drop(columns='Date')
        self.X_test = self.X_test.drop(columns='Date')

        print("\nTemporal split per Store/Dept complete.")
        print(f"Training samples: {self.X_train.shape[0]}  |  Test samples: {self.X_test.shape[0]}")

    def train(self):
        if not hasattr(self, "X_train_scaled"):
           print("Data not scaled yet. Scaling now")
           self.scale_data()
        start = time.time()
        self.model.fit(self.X_train_scaled, self.y_train)
        elapsed = time.time() - start
        print(f"training complete. Elapsed time: {elapsed:.3f} s")

    def prediction(self):
        return self.model.predict(self.X_test_scaled)

    def evaluate(self, plot=False):
        y_pred = self.model.predict(self.X_test_scaled)
        mae = mean_absolute_error(self.y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
        r2 = r2_score(self.y_test, y_pred)
        print(f"{type(self.model).__name__} Evaluation Metrics:")
        print(f"MAE: {mae:.3f}")
        print(f"RMSE: {rmse:.3f}")
        print(f"R²: {r2:.3f}")
        if plot:
            self.plots(y_pred)

    def plots(self, y_pred):
        sns.set_theme(style="whitegrid")

        plt.scatter(self.y_test, y_pred, alpha=0.6)
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.title("Predicted vs Actual")
        plt.plot([self.y_test.min(), self.y_test.max()], [self.y_test.min(), self.y_test.max()], 'r--')
        plt.show()

        residuals = self.y_test - y_pred
        plt.scatter(y_pred, residuals, alpha=0.6)
        plt.axhline(0, color='r', linestyle='--')
        plt.xlabel("Predicted values")
        plt.ylabel("Residuals")
        plt.title("Residual Plot")
        plt.show()

    def scale_data(self):
        scaler = StandardScaler()

        numeric_cols = self.X_train.select_dtypes(include=['float64', 'int64']).columns.tolist()
        exclude = ['Store', 'Dept', 'Day_sin', 'Day_cos', 'Month_sin', 'Month_cos', 'Years_since_start']
        numeric_cols = [c for c in numeric_cols if c not in exclude]

        print("Numerical columns to be standardized:", numeric_cols)
        self.X_train_scaled = self.X_train.copy()
        self.X_test_scaled = self.X_test.copy()
        self.X_train_scaled[numeric_cols] = scaler.fit_transform(self.X_train[numeric_cols])
        self.X_test_scaled[numeric_cols] = scaler.transform(self.X_test[numeric_cols])
        # print confirmation message
        print("Data scaling complete.")

# Linear models
class LinReg(RegressionModel):
    def __init__(self, dataset):
        super().__init__(dataset)
        self.model = LinearRegression()
        print("\n---------Linear Regression---------")

class RidgeModel(RegressionModel):
    def __init__(self, dataset, alpha=1.0):
        super().__init__(dataset)
        self.model = SKRidge(alpha)
        print("\n---------Ridge Regression---------")

class LassoModel(RegressionModel):
    def __init__(self, dataset, alpha=1.0):
        super().__init__(dataset)
        self.model = Lasso(alpha)
        print("\n---------Lasso Regression---------")

class ElasticNetModel(RegressionModel):
    def __init__(self, dataset, alpha=1.0, l1_ratio=0.5, random_state=42):
        super().__init__(dataset)
        self.model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=random_state)
        print("\n---------ElasticNet Regression---------")

# Non-linear tree based models
class DecisionTreeRegressorModel(RegressionModel):
    def __init__(self, dataset, max_depth=None, random_state=42):
        super().__init__(dataset)
        self.model = DecisionTreeRegressor(max_depth=max_depth, random_state=random_state)
        print("\n---------Decision Tree Regressor---------")

class RandomForestRegressorModel(RegressionModel):
    def __init__(self, dataset, n_estimators=100, random_state=42):
        super().__init__(dataset)
        self.model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
        print("\n---------Random Forest Regressor---------")

class XGBoostRegressorModel(RegressionModel):
    def __init__(self, dataset, n_estimators=100, learning_rate=0.1, random_state=42):
        super().__init__(dataset)
        self.model = XGBRegressor(n_estimators=n_estimators, learning_rate=learning_rate, random_state=random_state,
                                  enable_categorical=True, tree_method="hist")
        print("\n---------XGBoost Regressor---------")

# Autoregressive models
class ARIMAModel(RegressionModel):
    def __init__(self, dataset, order=(2, 1, 2)):
        super().__init__(dataset)
        self.order = order
        self.model_fitted = None

    def split_data(self, test_size=0.2, random_state=42, target_column="Weekly_Sales"):
        """Override: ARIMA needs a contiguous time-based split."""
        if target_column:
            self.target_column = target_column

        y = self.dataset.data[self.target_column].values
        n = len(y)
        split_point = int(n * (1 - test_size))
        self.y_train, self.y_test = y[:split_point], y[split_point:]
        print(f"\nARIMA time-based split: train={len(self.y_train)}, test={len(self.y_test)}")

    def train(self):
        """Override: fit ARIMA to training series."""
        print(f"Training ARIMA{self.order} model...")
        start = time.time()
        self.model = ARIMA(self.y_train, order=self.order)
        self.model_fit = self.model.fit()
        elapsed = time.time() - start
        print(f"Training complete in {elapsed:.3f}s")

    def prediction(self):
        """Override: forecast the same number of test points."""
        forecast = self.model_fit.forecast(steps=len(self.y_test))
        return forecast

    def evaluate(self, plots=False):
        y_pred = self.prediction()
        mae = mean_absolute_error(self.y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
        r2 = r2_score(self.y_test, y_pred)
        print(f"ARIMA{self.order} Evaluation Metrics:")
        print(f"MAE: {mae:.3f}")
        print(f"RMSE: {rmse:.3f}")
        print(f"R²: {r2:.3f}")
        if plots:
            self.plots(y_pred)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM as KerasLSTM, Dense, Input

class NeuralNetwork:
    def __init__(self, dataset):
        self.dataset = dataset
        self.model = None

    def split_data(self, target_column, group_cols=['Store', 'Dept'], test_size=0.2):
        X_train_list, X_test_list, y_train_list, y_test_list = [], [], [], []
        df = self.dataset.data.copy().sort_values(group_cols + ['Date'])

        for _, group in df.groupby(group_cols, observed=True):
            y = group[target_column].values
            X = group.drop(columns=[target_column])

            split_index = int(len(group) * (1 - test_size))
            X_train_list.append(X.iloc[:split_index])
            X_test_list.append(X.iloc[split_index:])
            y_train_list.append(y[:split_index])
            y_test_list.append(y[split_index:])

        self.X_train = pd.concat(X_train_list)
        self.X_test = pd.concat(X_test_list)
        self.y_train = np.concatenate(y_train_list)
        self.y_test = np.concatenate(y_test_list)

        self.X_train = self.X_train.drop(columns='Date')
        self.X_test = self.X_test.drop(columns='Date')

        print("\n Temporal split per Store/Dept complete.")
        print(f"Training samples: {self.X_train.shape[0]}  |  Test samples: {self.X_test.shape[0]}")
    
    def scale_data(self):
        x_scaler = MinMaxScaler()
        y_scaler = MinMaxScaler()

        numeric_cols = self.X_train.select_dtypes(include=['float64', 'int64']).columns.tolist()
        #exclude = ['Store', 'Dept', 'Day_sin', 'Day_cos', 'Month_sin', 'Month_cos', 'Years_since_start']
        #numeric_cols = [c for c in numeric_cols if c not in exclude]

        numeric_cols = [c for c in numeric_cols]

        print("Numerical columns to be standardized:", numeric_cols)
        self.X_train_scaled = self.X_train.copy()
        self.X_test_scaled = self.X_test.copy()
        self.X_train_scaled[numeric_cols] = x_scaler.fit_transform(self.X_train[numeric_cols])
        self.X_test_scaled[numeric_cols] = x_scaler.transform(self.X_test[numeric_cols])

        self.y_train_scaled = y_scaler.fit_transform(self.y_train.reshape(-1, 1)).flatten()
        self.y_test_scaled = y_scaler.transform(self.y_test.reshape(-1, 1)).flatten()

        self.x_scaler = x_scaler
        self.y_scaler = y_scaler

        # print confirmation message
        print("Data scaling complete.")
    
    def train(self, epochs=1000, batch_size=32, X_seq=None, y_seq=None, patience=10):
        if not hasattr(self, "X_train_scaled"):
            print("Data not scaled yet. Scaling now")
            self.scale_data()
        if X_seq is not None and y_seq is not None:
            X_train = X_seq
            y_train = y_seq
        else:
            X_train = np.asarray(self.X_train_scaled, dtype=np.float32)
            y_train = np.asarray(self.y_train_scaled, dtype=np.float32)

        early_stop = EarlyStopping(
            monitor='loss',       
            patience=patience,   
            restore_best_weights=True,
            verbose=1,
        )

        start = time.time()
        self.history = self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size ,verbose=2, callbacks=[early_stop])
        elapsed = time.time() - start
        print(f"training complete. Elapsed time: {elapsed:.3f} s")
        self.plot_training_history(self.history)
        return self.history
    
    def prediction(self):
        return self.model.predict(self.X_test_scaled)

    def evaluate(self):
        y_pred_scaled = self.model.predict(self.X_test_scaled).reshape(-1)

        mae  = mean_absolute_error(self.y_test_scaled, y_pred_scaled)
        rmse = np.sqrt(mean_squared_error(self.y_test_scaled, y_pred_scaled))
        r2   = r2_score(self.y_test_scaled, y_pred_scaled)

        mae = mae * (self.y_train.max() - self.y_train.min()) + self.y_train.min()
        rmse = rmse * (self.y_train.max() - self.y_train.min()) + self.y_train.min()
        

        print(f"{type(self.model).__name__} scaled Evaluation:")
        print(f"MAE:  {mae:.4f}") 
        print(f"RMSE: {rmse:.4f}")
        print(f"R²:   {r2:.4f}")

        y_pred_scaled = np.nan_to_num(y_pred_scaled, nan=0.0, posinf=10.0, neginf=-10.0)

        # Decide range based on scaler type
        if hasattr(self.y_scaler, "feature_range"):  # MinMaxScaler
            lo, hi = self.y_scaler.feature_range
            y_pred_scaled = np.clip(y_pred_scaled, lo, hi)
        elif hasattr(self.y_scaler, "scale_"):       # StandardScaler
            y_pred_scaled = np.clip(y_pred_scaled, -5, 5)  # ≈±5σ covers >99.9999% of data

        # --- Convert back to original units ---
        y_pred = y_pred_scaled * (self.y_train.max() - self.y_train.min()) + self.y_train.min()

        if not np.all(np.isfinite(y_pred)):
            print("Non-finite values still found after inverse scaling. Check scaling pipeline.")

        self.plots(y_pred)
    '''

    def evaluate(self):
        y_pred = self.model.predict(self.X_test_scaled).reshape(-1)

        mae  = mean_absolute_error(self.y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
        r2   = r2_score(self.y_test, y_pred)

        print(f"{type(self.model).__name__} scaled Evaluation:")
        print(f"MAE:  {mae:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"R²:   {r2:.4f}")

        self.plots(y_pred)
    '''

    def plot_training_history(self, history):
        """Plot loss / val_loss from a Keras History (or dict with 'loss'/'val_loss')."""
        if hasattr(history, "history"):
            h = history.history
        else:
            h = history
        plt.figure(figsize=(6,4))
        plt.plot(h.get('loss', []), label='train loss')
        if 'val_loss' in h:
            plt.plot(h['val_loss'], label='val loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training loss')
        plt.show()

    def plots(self, y_pred):
        """Visualize model performance with prediction vs actual and residual plots."""
        sns.set(style="whitegrid")

        y_true = np.array(self.y_test)
        residuals = y_true - y_pred.flatten()

        fig, axes = plt.subplots(1, 3, figsize=(16, 4))

        axes[0].scatter(y_true, y_pred, alpha=0.6, edgecolor='k')
        axes[0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
        axes[0].set_xlabel("Actual")
        axes[0].set_ylabel("Predicted")
        axes[0].set_title("Predicted vs Actual")

        axes[1].scatter(y_pred, residuals, alpha=0.6, edgecolor='k')
        axes[1].axhline(0, color='r', linestyle='--')
        axes[1].set_xlabel("Predicted values")
        axes[1].set_ylabel("Residuals")
        axes[1].set_title("Residuals vs Predicted")

        sns.histplot(residuals, bins=30, kde=True, ax=axes[2], color='steelblue')
        axes[2].set_xlabel("Residuals")
        axes[2].set_title("Distribution of Residuals")

        plt.tight_layout()
        plt.show()

class FeedforwardNN(NeuralNetwork):
    def __init__(self, dataset):
        super().__init__(dataset)
        n_cols = self.dataset.data.shape[1] - 2  # exclude target and Date
        # create model
        #opt = Adam(learning_rate=1e-3, clipnorm=1.0)
        self.model = Sequential()
        self.model.add(Input(shape=(n_cols,)))
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dense(1))
        # compile model
        self.model.compile(optimizer='adam', loss='mse')
        print("\n-----Feedforward Neural Network-----")
    
class LSTM(NeuralNetwork):
    def __init__(self, dataset, timesteps=5):
        super().__init__(dataset)
        self.timesteps = timesteps
        print("\n---------LSTM Neural Network---------")

    # Parent's split_data() will be called BEFORE sequence creation
    # so we no longer split inside this class.

    def prepare_sequences(self):
        """
        Convert tabular training and test data (from split_data)
        into 3D sequences (samples, timesteps, features).
        """
        def make_sequences(X_df, y_arr, timesteps):
            #X_df = X_df.select_dtypes(include=[np.number])  # drop non-numerics
            X_values = X_df.to_numpy(dtype=np.float32)
            X_seq, y_seq = [], []
            for i in range(timesteps, len(X_values)):
                X_seq.append(X_values[i - timesteps:i])
                y_seq.append(y_arr[i])
            return np.array(X_seq), np.array(y_seq)

        # Build sequences separately for train and test
        self.X_train_seq, self.y_train_seq = make_sequences(self.X_train, self.y_train, self.timesteps)
        self.X_test_seq, self.y_test_seq = make_sequences(self.X_test, self.y_test, self.timesteps)

        print(f"Train sequences: {self.X_train_seq.shape}, Test sequences: {self.X_test_seq.shape}")
        return self.X_train_seq, self.y_train_seq

    def scale_data(self, feature_range=(-1, 1)):
        """Apply MinMax scaling to 3D sequence data and targets."""
        feature_scaler = MinMaxScaler(feature_range=feature_range)
        target_scaler = MinMaxScaler(feature_range=feature_range)

        n_train, t, n_feat = self.X_train_seq.shape
        n_test = self.X_test_seq.shape[0]

        # Flatten for scaling
        X_train_2d = self.X_train_seq.reshape(-1, n_feat)
        X_test_2d = self.X_test_seq.reshape(-1, n_feat)

        # Fit on train, transform both
        self.X_train_scaled = feature_scaler.fit_transform(X_train_2d).reshape(n_train, t, n_feat)
        self.X_test_scaled = feature_scaler.transform(X_test_2d).reshape(n_test, t, n_feat)

        # Scale y separately
        self.y_train_scaled = target_scaler.fit_transform(self.y_train_seq.reshape(-1, 1)).flatten()
        self.y_test_scaled = target_scaler.transform(self.y_test_seq.reshape(-1, 1)).flatten()

        self.feature_scaler = feature_scaler
        self.target_scaler = target_scaler
        print(f"MinMax scaling done. Range: {feature_range}")

    def build_model(self):
        """Create and compile the LSTM model with an explicit Input layer."""
        n_features = self.X_train_scaled.shape[2]

        self.model = Sequential([
            Input(shape=(self.timesteps, n_features)),
            KerasLSTM(64, activation='tanh'),
            Dense(32, activation='relu'),
            Dense(1, activation='tanh')  # matches scaled y range (-1,1)
        ])

        self.model.compile(optimizer='adam', loss='mean_squared_error')
        self.model.summary()
        print("LSTM model compiled successfully.")

from tensorflow.keras import layers, models
from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization, Dropout, Dense, Input

class TransformerNN(NeuralNetwork):
    def __init__(self, dataset, timesteps=5, num_heads=4, ff_dim=64, dropout=0.1):
        super().__init__(dataset)
        self.timesteps = timesteps
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout = dropout

    def prepare_sequences(self, target_column):
        """create 3D sequences (samples, timesteps, features)."""
        data = self.dataset.data.copy().sort_values("Date")
        y = data[target_column].values
        X = data.drop(columns=[target_column]).values

        X_seq, y_seq = [], []
        for i in range(self.timesteps, len(X)):
            X_seq.append(X[i - self.timesteps:i])
            y_seq.append(y[i])

        X_seq = np.array(X_seq)
        y_seq = np.array(y_seq)
        print(f"Sequences created: {X_seq.shape}, Target: {y_seq.shape}")
        return X_seq, y_seq

    def build_model(self):
        """Create and compile a simple Transformer for regression."""
        n_features = self.X_train_scaled.shape[2]

        inputs = Input(shape=(self.timesteps, n_features))

        # --- Transformer Encoder Block ---
        attn_output = MultiHeadAttention(
            num_heads=self.num_heads, key_dim=n_features
        )(inputs, inputs)
        attn_output = Dropout(self.dropout)(attn_output)
        out1 = LayerNormalization(epsilon=1e-6)(inputs + attn_output)

        # Feed-forward network
        ffn = Dense(self.ff_dim, activation='relu')(out1)
        ffn = Dense(n_features)(ffn)
        ffn_output = Dropout(self.dropout)(ffn)
        out2 = LayerNormalization(epsilon=1e-6)(out1 + ffn_output)

        # --- Regression head ---
        x = layers.GlobalAveragePooling1D()(out2)
        x = Dense(64, activation='relu')(x)
        x = Dropout(self.dropout)(x)
        outputs = Dense(1)(x)

        self.model = models.Model(inputs=inputs, outputs=outputs)
        self.model.compile(optimizer='adam', loss='mean_squared_error')
        print(self.model.summary())


from Dataset import Dataset
def __main__():
    #dataset = Dataset("../dataset_cleaned_no_MarkDown_split_date.csv")

    # Print GPUs visible to TensorFlow
    print("TF GPUs:", tf.config.list_physical_devices('GPU'))

    dataset = Dataset("/mnt/c/Users/yurim/Documents/Work/Formazione_Experis/codice/Walmart/dataset_cleaned_no_MarkDown_split_date.csv")
    dataset.convert_nominal(columns=["Type"])
    dataset.drop_columns(columns=["Fuel_Price", "CPI", "Unemployment"])
    #dataset = pd.get_dummies(dataset, columns=["Type"], drop_first=True)
    #dataset = dataset.drop(columns=["Fuel_Price", "CPI", "Unemployment"])
    #droppato date
    #dataset = dataset.drop(columns=["Day", "Month", "Year"])

    lin_reg_model = LinReg(dataset)
    lin_reg_model.split_data(target_column="Weekly_Sales")
    lin_reg_model.train()
    lin_reg_model.evaluate()
    print(lin_reg_model.model.coef_)

    dataset =  Dataset("/mnt/c/Users/yurim/Documents/Work/Formazione_Experis/codice/Walmart/dataset_cleaned_no_data_noMD.csv")
    dataset.data["Date"] = pd.to_datetime(dataset.data["Date"])
    dataset.data = dataset.data.sort_values("Date")
    dataset.data = dataset.data.set_index("Date")

    arima_model = ARIMAModel(dataset, order=(1,1,1))
    arima_model.split_data(test_size=0.2, target_column="Weekly_Sales")
    arima_model.train()
    arima_model.evaluate(plots=True)

    '''
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

    ------------------------------------------------------------------------
    
    # LSTM model
    lstm_model = LSTM(dataset, timesteps=5)
    lstm_model.split_data(target_column="Weekly_Sales")

    X_seq, y_seq = lstm_model.prepare_sequences(target_column="Weekly_Sales")
    lstm_model.scale_data()

    lstm_model.build_model()
    lstm_model.train(epochs=50, batch_size=32)

    lstm_model.evaluate()
'''
if __name__ == "__main__":
    __main__()