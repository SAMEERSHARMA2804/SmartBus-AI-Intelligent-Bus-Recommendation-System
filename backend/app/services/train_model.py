import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib

# load dataset
df = pd.read_csv("data/routes.csv")

X = df[["distance","time_of_day","traffic_level","day_of_week"]]
y = df["eta"]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)

model=RandomForestRegressor()
model.fit(X_train,y_train)

joblib.dump(model,"model.pkl")

print("Model trained and saved")