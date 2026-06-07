# Pakistani-House-Price-Predictor

## Overview
A machine learning model that predicts house prices in Pakistan using real estate listing data. Built to explore Pakistan's housing market and provide data-driven price estimates. Best model: Random Forest with MAE of 0.86 crore PKR.

## Key Results
- **Best Model:** Random Forest
- **Mean Absolute Error:** 0.86 crore PKR (86 lakh rupees)
- **Cross-validation MAE:** 0.16 ± 0.01 (log scale)

## Dataset
- **Source:** (https://www.kaggle.com/datasets/diraf0/pakistan-housing-dataset?select=data.csv) 
- **Size:** ~7,500 rows after cleaning
- **Features:** 10 (area, beds, baths, location, amenities)

## Data Cleaning
- Price: Converted PKR/Crore/Lakh/Arab to numeric crore
- Area: Converted Marla/Kanal to kanal
- Location: Target encoding with smoothing (rare locations (locations with less than 10 houses) → global mean)
- Missing values: Median imputation for beds/baths, 0 for amenities
- Outliers: Removed houses >20 crore

## Feature Engineering
- Log transformation on target price
- Location encoding using average price per neighborhood
- Amenity features: gym, steam room, drawing room, kitchen count, store rooms, prayer rooms

## Models Tested
| Model | MAE (crore) |
|-------|-------------|
| Linear Regression | 7.23 |
| Random Forest | 0.88 |
| XGBoost | .89 |

**Winner:** Random Forest (lowest error, stable cross-validation)

## How to Run
1. Clone this repo
2. Install requirements: `pip install pandas numpy scikit-learn xgboost matplotlib`
3. Run `python main.py`

## Limitations
- Higher error on extreme prices (very cheap or very expensive houses)
- Limited by available features (no property age, condition, distance to amenities)
- Model is trained only on [your city/cities] data

## Future Improvements
- Add property age and condition features
- Build separate model for luxury houses (>15 crore)
    - Data seems to get skewed too much when there are too big of discrepencies between houses ie. 50 Lakh and 20 Crore
- Deploy as Streamlit web app
- Add more cities to training data

## Technologies Used
- Python (pandas, numpy, scikit-learn, xgboost)
- Random Forest, XGBoost, Linear Regression
- 5-fold cross-validation

## Author
Ismail Chaudry — Tulane University, Class of 2030

## Acknowledgments
- Data source: (https://www.kaggle.com/datasets/diraf0/pakistan-housing-dataset?select=data.csv)
- Inspired by Pakistan's real estate market
