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

print(df['Location'].nunique())
print(df['Location'].value_counts().head(10))


''' plt.scatter(df['Price_Crore'], df['Area_Kanal'])
plt.xlabel("Price in Crore")
plt.ylabel('Area in Kanal')
plt.title("Price vs Area")
plt.show() '''