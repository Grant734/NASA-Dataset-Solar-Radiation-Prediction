#if you want to run this code yourself, make sure you have all the correct downloads for the imports, and have downloaded the csv file from this website: https://www.kaggle.com/datasets/dronio/SolarEnergy/data
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

df = pd.read_csv("/Users/Grant/Desktop/SolarPrediction.csv")
print("Initial DataFrame (first 5 rows):")
print(df.head())



#Parsing Date/Time into numeric features
df["Datetime"] = pd.to_datetime(df["Data"] + " " + df["Time"])

df["day_of_year"] = df["Datetime"].dt.dayofyear
df["time_of_day"] = (
    df["Datetime"].dt.hour
    + df["Datetime"].dt.minute / 60.0
    + df["Datetime"].dt.second / 3600.0
)

df.drop(["Data", "Time", "Datetime"], axis=1, inplace=True)


#Parsing Sunrise/Sunset into numeric features
def parse_time_str(t_str):
    try:
        h, m, s = t_str.split(':')
        return float(h) + float(m) / 60.0 + float(s) / 3600.0
    except:
        return np.nan

df["TimeSunRise_float"] = df["TimeSunRise"].apply(parse_time_str)
df["TimeSunSet_float"] = df["TimeSunSet"].apply(parse_time_str)

df.drop(["TimeSunRise", "TimeSunSet"], axis=1, inplace=True)

#converting relevant columns to floats
df = df.astype(float)

print("DataFrame after parsing:")
print(df.head())


#defining features (x) and target (y)

df["Radiation_log"] = np.log1p(df["Radiation"])

feature_cols = [  "UNIXTime",
    "Temperature",
    "Pressure",
    "Humidity",
    "WindDirection(Degrees)",
    "Speed",
    "TimeSunRise_float",
    "TimeSunSet_float",
    "day_of_year",
    "time_of_day"] 

target_col = "Radiation_log"


#train/test split
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
print(f"Training set size: {len(train_df)}")
print(f"Test set size: {len(test_df)}")


#scaling the data
scaler_X = StandardScaler()
scaler_y = StandardScaler()

scaler_X.fit(train_df[feature_cols])
scaler_y.fit(train_df[[target_col]])

X_train_scaled = scaler_X.transform(train_df[feature_cols])
y_train_scaled = scaler_y.transform(train_df[[target_col]])

X_test_scaled = scaler_X.transform(test_df[feature_cols])
y_test_scaled = scaler_y.transform(test_df[[target_col]])



#creating a pytorch dataset class
class RadiationDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        X_data = torch.tensor(self.X[idx], dtype=torch.float32)
        y_data = torch.tensor(self.y[idx], dtype=torch.float32)
        return X_data, y_data

train_dataset = RadiationDataset(X_train_scaled, y_train_scaled)
test_dataset  = RadiationDataset(X_test_scaled,  y_test_scaled)


train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=32, shuffle=False)


#defining the feedforward model architecture
class SolarNet(nn.Module):
    def __init__(self, input_size, hidden_size=64):
        super(SolarNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

input_size = len(feature_cols)
net = SolarNet(input_size=input_size, hidden_size=64)
print(net)


#weighted MSE loss function
def weighted_mse_loss(preds, targets, alpha=0.2):
    with torch.no_grad():
        targets_np = targets.cpu().numpy()
        log_targets = scaler_y.inverse_transform(targets_np)
        rad_values = np.expm1(log_targets)
        rad_values_torch = torch.tensor(rad_values, dtype=torch.float32, device=preds.device)
        weights = 1.0 + alpha * rad_values_torch
    squared_diff = (preds - targets).pow(2)
    loss = (weights * squared_diff).mean()
    return loss


#loss function/optimizer

optimizer = optim.Adam(net.parameters(), lr=0.0005)

#moving model to GPU
device = torch.device("mps" if torch.backends.mps.is_available()
                      else "cuda" if torch.cuda.is_available()
                      else "cpu")
net.to(device)
print("Using device:", device)


#training loop
EPOCHS = 15

for epoch in range(EPOCHS):
    net.train()
    total_loss = 0.0
    
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        optimizer.zero_grad()
        outputs = net(X_batch)
        loss = weighted_mse_loss(outputs, y_batch, alpha=0.2)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{EPOCHS}], Weighted Loss: {avg_loss:.4f}")


#evaluating on test set
net.eval()
test_loss = 0.0

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        outputs = net(X_batch)
        loss = weighted_mse_loss(outputs, y_batch, alpha=0.2)
        test_loss += loss.item()

test_loss /= len(test_loader)
rmse_scaled = test_loss**0.5
print(f"\nTest Weighted MSE (scaled log target): {test_loss:.4f}")
print(f"Test RMSE (scaled log target):         {rmse_scaled:.4f}")

std_of_log_y = scaler_y.scale_[0]
rmse_log_space = rmse_scaled * std_of_log_y
rmse_original_est = np.expm1(rmse_log_space)

print(f"Approx. RMSE (original scale): {rmse_original_est:.4f}")


#saving the model and scalers

model_path = "solarnet_weights.pth"
torch.save(net.state_dict(), model_path)
print(f"Model weights saved to: {model_path}")

joblib.dump(scaler_X, "scaler_X.pkl")
joblib.dump(scaler_y, "scaler_y.pkl")
print("Scalers saved to: scaler_X.pkl, scaler_y.pkl")

#defining a function to load the model and scalers
def load_model_and_scalers():
    loaded_net = SolarNet(input_size=len(feature_cols), hidden_size=64)
    loaded_net.load_state_dict(torch.load("solarnet_weights.pth"))
    loaded_net.eval()
    loaded_net.to(device)

    loaded_scaler_X = joblib.load("scaler_X.pkl")
    loaded_scaler_y = joblib.load("scaler_y.pkl")

    return loaded_net, loaded_scaler_X, loaded_scaler_y

print("When inputing a date, do it by the date's number in the year (ex.: 270). When inputing a time, make it a float in this format: hour + minute/60 + second/3600.")

#user prediction
def predict_with_user_input(omit_column_name=None):
    expected_cols = feature_cols[:]  # make a copy

    loaded_net, loaded_scaler_X, loaded_scaler_y = load_model_and_scalers()
    user_features = []

    means_of_columns = dict()
    if hasattr(loaded_scaler_X, "mean_"):
        for col, mean_val in zip(feature_cols, loaded_scaler_X.mean_):
            means_of_columns[col] = mean_val
    else:
        for col in feature_cols:
            means_of_columns[col] = 0.0

    for col in expected_cols:
        if col == omit_column_name:
            print(f"Skipping {col} (omitted), using default = {means_of_columns[col]}")
            user_features.append(means_of_columns[col])
        else:
            val_str = input(f"Enter value for {col}: ")
            if val_str.strip() == "":
                print(f"No input provided. Using default = {means_of_columns[col]}")
                user_features.append(means_of_columns[col])
            else:
                val_float = float(val_str)
                user_features.append(val_float)

    X_new = np.array([user_features], dtype=np.float32)

    X_new_scaled = loaded_scaler_X.transform(X_new)

    X_new_torch = torch.tensor(X_new_scaled, dtype=torch.float32).to(device)

    with torch.no_grad():
        preds_scaled_log = loaded_net(X_new_torch)
    
    preds_scaled_log_np = preds_scaled_log.cpu().numpy()
    preds_log = loaded_scaler_y.inverse_transform(preds_scaled_log_np)

    preds_original = np.expm1(preds_log)

    preds_original += 1.5

    # printing result
    print("==========================================")
    print("User features (unscaled):", user_features)
    print("Predicted Radiation:", preds_original[0][0])
    print("==========================================") 

    
#Visualizing predictions versus true values
X_test, y_test = next(iter(test_loader))
X_test, y_test = X_test.to(device), y_test.to(device)

with torch.no_grad():
    preds_scaled = net(X_test).cpu().numpy()

y_test_scaled = y_test.cpu().numpy()

if __name__ == "__main__":
    print("\nNow you can enter custom inputs for Radiation prediction.\n")
    while True:
        predict_with_user_input()  # <-- rename here
        cont = input("Predict again? (y/n): ").strip().lower()
        if cont != "y":
            break