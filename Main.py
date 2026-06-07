import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

df = pd.read_csv('data.csv')

def clean_one_price(price_string):
    price_string = price_string.replace('PKR', '')
    if 'Lakh' in price_string:
        price_string = price_string.replace('Lakh','')
        price_string = price_string.strip()
        price_string = float(price_string)
        price_string = price_string/100
    elif "Crore" in price_string:
        price_string =price_string.replace('Crore','')
        price_string = price_string.strip()
        price_string = float(price_string)
    elif "Arab" in price_string:
        price_string = price_string.replace('Arab','')
        price_string = float(price_string)
        price_string = price_string * 100
    return price_string

def one_area(area_string):
    if 'Marla' in area_string:
        area_string = area_string.replace('Marla','')
        area_string =area_string.strip()
        area_string = float(area_string)
        area_string = area_string/20
    elif 'Kanal' in area_string:
        area_string = area_string.replace('Kanal','')
        area_string = area_string.strip()
        area_string = float(area_string)
    return area_string

df['Price_Crore'] = df['Price'].apply(clean_one_price)

df['Area_Kanal'] = df['Area'].apply(one_area)
df['Area_Kanal'] = pd.to_numeric(df['Area_Kanal'], errors='coerce')
df= df.dropna(subset=['Area_Kanal'])

X = df[['Area','Baths','Beds','Location']]
y = df['Price_Crore']

Location_mean = df['Location']['Price_Crore'].mean()
df['Location_encoded'] = df['Location'].map(Location_mean)
global_mean = df['Location_encoded'].mean()
df['Location_encoded'].fillna(global_mean)

model = LinearRegression()

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.2,random_state=42)

model.fit(X_train,y_train)
y_predict = model.predict(X_test)

print("Actual Price:", y_test.values)
print('Predicted Price:', y_predict)