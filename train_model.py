import numpy as np
from sklearn.linear_model import LinearRegression
import pickle

X = np.random.rand(100, 1) * 10
y = 3 * X.squeeze() + 7 + np.random.randn(100)

model = LinearRegression()
model.fit(X, y)

try:
    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)
    print("✅ Model trained and saved as model.pkl")
except Exception as e:
    print("❌ Failed to save the model:", e)

