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
        area_string = area_string.strip()
        area_string = float(area_string)
    elif 'Kanal' in area_string:
        area_string = area_string.replace('Kanal','')
        area_string =area_string.strip()
        area_string = float(area_string)
        area_string = area_string/20
    return area_string

df['Price_Crore'] = df['Price'].apply(clean_one_price)

df['Area_Marla'] = df['Area'].apply(one_area)

print(df['Area_Marla'].head(10))

# plt.scatter(df['Price_Crore'], df[])