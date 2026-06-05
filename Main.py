import pandas as pd

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

df['Price_Crore'] = df['Price'].apply(clean_one_price)

print(df['Price_Crore'].head(25))