import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
import numpy as np
import xgboost as xgb
from xgboost import XGBRegressor

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

df['Beds_numeric'] = pd.to_numeric(df['Beds'], errors='coerce')
beds_median = df['Beds_numeric'].median()
df['Beds_clean']= df['Beds_numeric'].fillna(beds_median)

df['Baths_numeric'] = pd.to_numeric(df['Baths'], errors='coerce')
baths_median = df['Baths_numeric'].median()
df['Baths_clean']= df['Baths_numeric'].fillna(baths_median)

global_mean = df['Price_Crore'].mean()
Location_mean = df.groupby('Location')['Price_Crore'].mean()
location_counts = df.groupby('Location')['Price_Crore'].count()
df['location_count'] = df['Location'].map(location_counts)

df['Location_encoded'] = np.where(df['location_count'] > 10, df['Location'].map(Location_mean), global_mean )

df_clean = df[df['Price_Crore'] <= 20]
df = df[df['Price_Crore'] > 0]
df_clean['log_price'] = np.log(df_clean['Price_Crore'])

X = df_clean[['Area_Kanal','Baths_clean','Beds_clean','Location_encoded','Gym','Dining Room','Kitchens']]
y = df_clean['log_price']

# model = LinearRegression(), MAE = 1.59, MSE = 7.23
model = RandomForestRegressor(n_estimators=300,max_depth=15, random_state=42) # MAE = .88, MSE = 2.17 (Lowest)
# model = XGBRegressor(n_estimators=300,max_depth=15, random_state=42) # MAE = .89, MSE = 2.32

cv_scores = cross_val_score(model, X, y, cv=5,scoring='neg_mean_absolute_error')
mae_scores = -cv_scores
print(f"CV MAE: {mae_scores.mean():.2f} (+/- {mae_scores.std():.2f})")

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.2,random_state=42)

model.fit(X_train,y_train)
log_predict = model.predict(X_test)
y_predict = np.exp(log_predict)

print('Actual Price (in Crore):', [f'{x:.2f}' for x in np.exp(y_test[:10])],)
print('Predicted Price (in Crore):', y_predict[:10])

y_test_crore = np.exp(y_test)
mae = mean_absolute_error(y_test_crore, y_predict)
mse = mean_squared_error(y_test_crore, y_predict)

print(f'Mean Absolute Error {mae:.2f}')
print(f'Mean Squared Error {mse:.2f}')

plt.scatter(y_test_crore, y_predict, alpha=0.5)
plt.plot([0, 30], [0, 30], 'r--', label='Perfect prediction')
plt.xlabel('Actual Price (Crore)')
plt.ylabel('Predicted Price (Crore)')
plt.title('Random Forest: Actual vs Predicted')
plt.legend()
plt.savefig('predictions_plot.png', dpi=150, bbox_inches='tight')




