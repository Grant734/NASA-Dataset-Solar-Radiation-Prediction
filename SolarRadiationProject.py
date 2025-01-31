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

# Parsing Date/Time into numeric features
df["Datetime"] = pd.to_datetime(df["Data"] + " " + df["Time"])
df["day_of_year"] = df["Datetime"].dt.dayofyear
df["time_of_day"] = (
    df["Datetime"].dt.hour + df["Datetime"].dt.minute / 60.0 + df["Datetime"].dt.second / 3600.0
)
df.drop(["Data", "Time", "Datetime"], axis=1, inplace=True)

# Parsing Sunrise/Sunset into numeric features
def parse_time_str(t_str):
    try:
        h, m, s = t_str.split(':')
        return float(h) + float(m) / 60.0 + float(s) / 3600.0
    except:
        return np.nan

df["TimeSunRise_float"] = df["TimeSunRise"].apply(parse_time_str)
df["TimeSunSet_float"] = df["TimeSunSet"].apply(parse_time_str)
df.drop(["TimeSunRise", "TimeSunSet"], axis=1, inplace=True)

# Converting relevant columns to floats
df = df.astype(float)
print("DataFrame after parsing:")
print(df.head())

# Defining features (X) and target (y)
feature_cols = [
    "UNIXTime", "Temperature", "Pressure", "Humidity", "WindDirection(Degrees)",
    "Speed", "TimeSunRise_float", "TimeSunSet_float", "day_of_year", "time_of_day"
]
target_col = "Radiation"

# Train/test split
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
print(f"Training set size: {len(train_df)}")
print(f"Test set size: {len(test_df)}")

# Scaling the data
scaler_X = StandardScaler()
scaler_y = StandardScaler()

scaler_X.fit(train_df[feature_cols])
scaler_y.fit(train_df[[target_col]])

X_train_scaled = scaler_X.transform(train_df[feature_cols])
y_train_scaled = scaler_y.transform(train_df[[target_col]])

X_test_scaled = scaler_X.transform(test_df[feature_cols])
y_test_scaled = scaler_y.transform(test_df[[target_col]])

# Creating a PyTorch dataset class
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


# Defining the feedforward model architecture
class SolarNet(nn.Module):
    def __init__(self, input_size, hidden_size=64):
        super(SolarNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, hidden_size)
        self.fc5 = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x


input_size = len(feature_cols)
#net = SolarNet(input_size=input_size, hidden_size=64)
#print(net)


"""
#list of learning rates to test
learning_rates = [0.001, 0.0005, 0.0001, 0.00005, 0.00001]
"""

learning_rate = 0.001

"""
#list of batch sizes to test
batch_sizes = [16, 32, 64, 128, 256]
results = {}
"""
batch_size = 16

# Moving model to GPU
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

train_loader = DataLoader(RadiationDataset(X_train_scaled, y_train_scaled), batch_size=batch_size, shuffle=True)
test_loader = DataLoader(RadiationDataset(X_test_scaled, y_test_scaled), batch_size=batch_size, shuffle=False)

print("\nTraining with 5 layers")
net = SolarNet(input_size=input_size, hidden_size=64).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=learning_rate)
train_losses = []
test_losses = []

for epoch in range(8):
    net.train()
    total_loss = 0.0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = net(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    train_losses.append(total_loss / len(train_loader))
    
    net.eval()
    test_loss = 0.0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = net(X_batch)
            loss = criterion(outputs, y_batch)
            test_loss += loss.item()
    test_losses.append(test_loss / len(test_loader))
    print(f"Epoch {epoch+1}: Train Loss = {train_losses[-1]:.4f}, Test Loss = {test_losses[-1]:.4f}")

#defining a function to load the model and scalers
def load_model_and_scalers():
    loaded_net = SolarNet(input_size=len(feature_cols), hidden_size=64)
    loaded_net.load_state_dict(torch.load("solarnet_weights.pth"))
    loaded_net.eval()  
    loaded_net.to(device)  

    loaded_scaler_X = joblib.load("scaler_X.pkl")
    loaded_scaler_y = joblib.load("scaler_y.pkl")

    return loaded_net, loaded_scaler_X, loaded_scaler_y

# Plotting results
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, linestyle='dashed', label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Testing Loss for 5-Layer Neural Network')
plt.legend()
plt.show()

"""
# Saving the model and scalers
model_path = "solarnet_weights.pth"
torch.save(net.state_dict(), model_path)
print(f"Model weights saved to: {model_path}")
joblib.dump(scaler_X, "scaler_X.pkl")
joblib.dump(scaler_y, "scaler_y.pkl")
print("Scalers saved to: scaler_X.pkl, scaler_y.pkl")
"""


print("When inputing a date, do it by the date's number in the year (ex.: 270). When inputing a time, make it a float in this format: hour + minute/60 + second/3600.")

#user prediction
def predict_with_user_input():
    loaded_net, loaded_scaler_X, loaded_scaler_y = load_model_and_scalers()
    user_features = []

    for col in feature_cols:
        val_str = input(f"Enter value for {col}: ")
        if val_str.strip() == "":
            print("Invalid input, using mean value.")
            user_features.append(loaded_scaler_X.mean_[feature_cols.index(col)])
        else:
            user_features.append(float(val_str))

    X_new = np.array([user_features], dtype=np.float32)
    X_new_scaled = loaded_scaler_X.transform(X_new)
    X_new_torch = torch.tensor(X_new_scaled, dtype=torch.float32).to(device)

    with torch.no_grad():
        prediction_scaled = loaded_net(X_new_torch).cpu().numpy()
    prediction = loaded_scaler_y.inverse_transform(prediction_scaled)

    # printing result
    print("==========================================")
    print("User features (unscaled):", user_features)
    print("Predicted Radiation:", prediction[0][0])
    print("==========================================")

    
#Visualizing predictions versus true values
X_test, y_test = next(iter(test_loader))
X_test, y_test = X_test.to(device), y_test.to(device)

with torch.no_grad():
    preds_scaled = net(X_test).cpu().numpy()

y_test_scaled = y_test.cpu().numpy()

preds_original = scaler_y.inverse_transform(preds_scaled)
y_test_original = scaler_y.inverse_transform(y_test_scaled)

for i in range(min(5, len(preds_original))):  
    print(f"Actual: {y_test_original[i][0]:.2f}, Predicted: {preds_original[i][0]:.2f}")

if __name__ == "__main__":
    print("\nNow you can enter custom inputs for Radiation prediction.\n")
    while True:
        predict_with_user_input()
        cont = input("Predict again? (y/n): ").strip().lower()
        if cont != "y":
            break