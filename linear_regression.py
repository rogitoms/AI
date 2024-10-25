import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


file_path = '../AI/Nairobi Office Price Ex.csv'
data = pd.read_csv(file_path)

X = data['SIZE'].values  
y = data['PRICE'].values  

def compute_mse(y_true, y_pred):
    mse = np.mean((y_true - y_pred) ** 2)
    return mse

def gradient_descent(X, y, learning_rate=0.0001, epochs=10):
    m = np.random.rand()  
    c = np.random.rand()  
    n = len(X)  
    errors = []  

    for epoch in range(epochs):
        y_pred = m * X + c  
        error = compute_mse(y, y_pred)  
        errors.append(error)

        dm = -(2/n) * np.sum(X * (y - y_pred))  
        dc = -(2/n) * np.sum(y - y_pred)       
        
        m -= learning_rate * dm
        c -= learning_rate * dc

        print(f"Epoch {epoch+1}/{epochs}, MSE: {error}")
    
    return m, c, errors

m, c, errors = gradient_descent(X, y, learning_rate=0.0001, epochs=10)

plt.scatter(X, y, color='blue', label='Data Points')  
plt.plot(X, m * X + c, color='red', label='Line of Best Fit')  
plt.xlabel('Office Size (sq. ft.)')
plt.ylabel('Office Price')
plt.title('Line of Best Fit after 10 Epochs')
plt.legend()
plt.show()

predicted_price = m * 100 + c
print(f"Predicted price for 100 sq. ft. office: {predicted_price}")
