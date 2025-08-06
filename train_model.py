import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

# Sample training data
data = {'hours': [1, 2, 3, 4, 5], 'marks': [20, 40, 60, 80, 100]}
df = pd.DataFrame(data)

X = df[['hours']]
y = df[['marks']]

# Train model
model = LinearRegression()
model.fit(X, y)

# Save model
joblib.dump(model, 'student_mark_predictor.pkl')

print("✅ Model trained and saved as student_mark_predictor.pkl")
