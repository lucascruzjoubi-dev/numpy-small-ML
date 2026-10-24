import numpy as np 

X = np.array([[50, 1], [80, 2], [100, 3], [120, 3], [150, 4]], dtype=float)
y = np.array([165, 260, 325, 385, 480], dtype=float)


X_mean = np.mean(X, axis=0)
X_std = np.std(X, axis=0)
X_norm = (X - X_mean) / X_std

n_samples, n_characteristics = X_norm.shape
W = np.zeros(n_characteristics)
b = 0.0


learning_rate = 0.01
epochs = 1000


for epoch in range(epochs):
    y_predicted = np.dot(X_norm, W) + b 

    error = y_predicted - y

    dw = (2 / n_samples) * np.dot(X_norm.T, error)
    db = (2 / n_samples) * np.sum(error)

    W = W - learning_rate * dw
    b = b - learning_rate * db



print('--- Training Finished---')
print(f'Final weights (W): {W}')
print(f'Final bias (b): {b:4f}\n')

new_property = np.array([110, 3], dtype=float)
new_property_norm = (new_property - X_mean) / X_std

prediction = np.dot(new_property_norm, W) + b
print(f'Prediction for 110 m² and 3 bedrooms: R$ {prediction:2f}')


