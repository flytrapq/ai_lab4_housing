import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('housing.csv')

# Encode categorical variable (ocean_proximity)
label_encoder = LabelEncoder()
df['ocean_proximity_encoded'] = label_encoder.fit_transform(df['ocean_proximity'])
df.drop('ocean_proximity', axis=1, inplace=True)

# Split the dataset into features (X) and target variable (y)
X = df.drop(columns=['median_house_value'])
y = df['median_house_value']

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the scaled data into training and testing sets
X_train_scaled, X_test_scaled, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Handle missing values
imputer = SimpleImputer(strategy='mean')
X_train_imputed = imputer.fit_transform(X_train_scaled)
X_test_imputed = imputer.transform(X_test_scaled)

# Reshape the input training data for RNN to include a "fake" time dimension
X_train_imputed_reshaped = X_train_imputed.reshape(X_train_imputed.shape[0], 1, X_train_imputed.shape[1])
X_test_imputed_reshaped = X_test_imputed.reshape(X_test_imputed.shape[0], 1, X_test_imputed.shape[1])

# Define and train the RNN model
model_rnn = Sequential([
    SimpleRNN(units=32, activation='relu', input_shape=(1, X_train_imputed_reshaped.shape[2])),
    Dense(units=1)
])
model_rnn.compile(optimizer=Adam(), loss='mean_squared_error')
history = model_rnn.fit(X_train_imputed_reshaped, y_train, epochs=100, batch_size=32, validation_split=0.2)

# Evaluate the RNN model
y_pred_rnn = model_rnn.predict(X_test_imputed_reshaped)

# Plot training and validation loss over epochs
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

# Calculate the range of the target variable
target_range = y.max() - y.min()

# Evaluate the performance of each model
models = {
    'Linear Regression': {'model': LinearRegression(), 'color': 'blue'},
    'Polynomial Regression': {'model': LinearRegression(), 'color': 'green'},
    'Random Forest': {'model': RandomForestRegressor(n_estimators=100, random_state=42), 'color': 'red'},
    'RNN': {'model': model_rnn, 'color': 'orange'}
}

fig, axes = plt.subplots(1, 4, figsize=(24, 6))

for i, (name, model_info) in enumerate(models.items()):
    model = model_info['model']
    color = model_info['color']
    
    if name == 'Polynomial Regression':
        poly_features = PolynomialFeatures(degree=2)
        X_train_poly = poly_features.fit_transform(X_train_imputed)
        X_test_poly = poly_features.transform(X_test_imputed)
        model.fit(X_train_poly, y_train)
        y_pred = model.predict(X_test_poly)
    elif name == 'RNN':
        y_pred = y_pred_rnn.flatten()
    else:
        model.fit(X_train_imputed, y_train)
        y_pred = model.predict(X_test_imputed).flatten()
    
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    
    # Normalize MAE and MSE to the target variable range
    mae_percent = (mae / target_range) * 100
    mse_percent = (mse / (target_range ** 2)) * 100
    
    print(f"{name}:")
    print(f"Mean Absolute Error: {mae_percent:.2f}%")
    print(f"Mean Squared Error: {mse_percent:.2f}%")
    print()
    
    axes[i].scatter(y_test, y_pred, color=color, label=name)
    axes[i].plot(y_test, y_test, color='black', label='Actual Prices')
    axes[i].set_title(name)
    axes[i].set_xlabel('Actual Values')
    axes[i].set_ylabel('Predicted Values')
    axes[i].legend()

plt.tight_layout()
plt.show()