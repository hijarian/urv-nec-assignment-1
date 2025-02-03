from NeuralNet import NeuralNet
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

scaler = StandardScaler()

def load_data(filename, columns_to_normalize=None):
    data = pd.read_csv(filename)
    data = data.astype(float)

    data[columns_to_normalize] = scaler.fit_transform(data[columns_to_normalize])

    data['Revenue'] = scaler.fit_transform(data['Revenue'].values.reshape(-1, 1))

    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

    X_train = train_data.drop(columns=['Revenue']).values
    y_train = train_data['Revenue'].values

    X_test = test_data.drop(columns=['Revenue']).values
    y_test = test_data['Revenue'].values

    return X_train, y_train, X_test, y_test

def main():
    X_train, y_train, X_test, y_test = load_data(
        'data/reduced_dataset.csv',
        columns_to_normalize=['Seating Capacity', 'Average Meal Price', 'Weekend Reservations', 'Weekday Reservations']
    )

    input_layer = X_train.shape[1]
    layers = [input_layer, 27, 17, 1]

    nn = NeuralNet(layers=layers,
                    epochs=30,
                      learning_rate=0.001,
                        momentum=0.06, 
                        activation_function_name='linear', 
                        validation_split=0.2
                    , visualize=None
                   )
    nn.fit(X_train, y_train)

    y_pred = nn.predict(X_test)

    # Calculate the mean squared error
    mse = mean_squared_error(y_test, y_pred)

    # Calculate the mean absolute error
    mae = mean_absolute_error(y_test, y_pred)

    # Calculate the mean absolute percentage error
    mape = mean_absolute_percentage_error(y_test, y_pred)

    print(f"Mean Squared Error: {mse}")
    print(f"Mean Absolute Error: {mae}")
    print(f"Mean Absolute Percentage Error: {mape}%")

    plt.plot(nn.training_errors, label='Training Error')
    plt.plot(nn.validation_errors, label='Validation Error')
    plt.xlabel('Epochs')
    plt.ylabel('Error')
    plt.title('Training and Validation Error over Epochs')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()
