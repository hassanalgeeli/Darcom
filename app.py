import pandas as pd
from flask import Flask, request, jsonify, render_template
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor


data = [
    {"Region": "North Riyadh", "Neighborhood": "Al Malqa", "Property_Type": "Villa", "Area_sqm": 450, "Bedrooms": 5, "Property_Age_Years": 2, "Price_SAR": 3200000},
    {"Region": "South Riyadh", "Neighborhood": "Al Shifa", "Property_Type": "Apartment", "Area_sqm": 180, "Bedrooms": 3, "Property_Age_Years": 7, "Price_SAR": 750000},
    {"Region": "East Riyadh", "Neighborhood": "Al Rimal", "Property_Type": "Duplex", "Area_sqm": 280, "Bedrooms": 4, "Property_Age_Years": 3, "Price_SAR": 1400000},
    {"Region": "West Riyadh", "Neighborhood": "Laban", "Property_Type": "Villa", "Area_sqm": 350, "Bedrooms": 4, "Property_Age_Years": 10, "Price_SAR": 1800000},
    {"Region": "North Riyadh", "Neighborhood": "Al Yasmeen", "Property_Type": "Apartment", "Area_sqm": 210, "Bedrooms": 3, "Property_Age_Years": 1, "Price_SAR": 1100000},
    {"Region": "South Riyadh", "Neighborhood": "Al Dar Al Baida", "Property_Type": "Villa", "Area_sqm": 400, "Bedrooms": 5, "Property_Age_Years": 12, "Price_SAR": 1300000},
    {"Region": "East Riyadh", "Neighborhood": "Al Naseem", "Property_Type": "Apartment", "Area_sqm": 160, "Bedrooms": 2, "Property_Age_Years": 20, "Price_SAR": 550000},
    {"Region": "West Riyadh", "Neighborhood": "Al Uraija", "Property_Type": "Duplex", "Area_sqm": 250, "Bedrooms": 4, "Property_Age_Years": 5, "Price_SAR": 1250000},
    {"Region": "North Riyadh", "Neighborhood": "Al Narjis", "Property_Type": "Villa", "Area_sqm": 500, "Bedrooms": 6, "Property_Age_Years": 0, "Price_SAR": 4500000},
    {"Region": "South Riyadh", "Neighborhood": "Al Aziziyah", "Property_Type": "Apartment", "Area_sqm": 175, "Bedrooms": 3, "Property_Age_Years": 8, "Price_SAR": 680000},
    {"Region": "East Riyadh", "Neighborhood": "Al Qadisiyah", "Property_Type": "Villa", "Area_sqm": 380, "Bedrooms": 4, "Property_Age_Years": 1, "Price_SAR": 2100000},
    {"Region": "West Riyadh", "Neighborhood": "Al Badiah", "Property_Type": "Apartment", "Area_sqm": 190, "Bedrooms": 3, "Property_Age_Years": 15, "Price_SAR": 720000},
    {"Region": "North Riyadh", "Neighborhood": "Al Sahafah", "Property_Type": "Duplex", "Area_sqm": 300, "Bedrooms": 4, "Property_Age_Years": 4, "Price_SAR": 2300000},
    {"Region": "South Riyadh", "Neighborhood": "Al Shifa", "Property_Type": "Villa", "Area_sqm": 420, "Bedrooms": 5, "Property_Age_Years": 6, "Price_SAR": 1500000},
    {"Region": "East Riyadh", "Neighborhood": "Al Rimal", "Property_Type": "Apartment", "Area_sqm": 200, "Bedrooms": 3, "Property_Age_Years": 2, "Price_SAR": 900000},
    {"Region": "West Riyadh", "Neighborhood": "Laban", "Property_Type": "Villa", "Area_sqm": 390, "Bedrooms": 5, "Property_Age_Years": 9, "Price_SAR": 1950000},
    {"Region": "North Riyadh", "Neighborhood": "Al Malqa", "Property_Type": "Apartment", "Area_sqm": 220, "Bedrooms": 3, "Property_Age_Years": 1, "Price_SAR": 1200000},
    {"Region": "South Riyadh", "Neighborhood": "Al Dar Al Baida", "Property_Type": "Duplex", "Area_sqm": 260, "Bedrooms": 4, "Property_Age_Years": 11, "Price_SAR": 950000},
    {"Region": "East Riyadh", "Neighborhood": "Al Naseem", "Property_Type": "Villa", "Area_sqm": 500, "Bedrooms": 6, "Property_Age_Years": 22, "Price_SAR": 1800000},
    {"Region": "West Riyadh", "Neighborhood": "Al Uraija", "Property_Type": "Villa", "Area_sqm": 410, "Bedrooms": 5, "Property_Age_Years": 4, "Price_SAR": 2050000}
]

df = pd.DataFrame( data)


def train_model():
    X = df.drop('Price_SAR', axis=1)
    y = df['Price_SAR']
    categorical_features = ['Region', 'Neighborhood', 'Property_Type']
    numerical_features = ['Area_sqm', 'Bedrooms', 'Property_Age_Years']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])

    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', RandomForestRegressor(n_estimators=10, random_state=42))
    ])

    model_pipeline.fit(X, y )
    print("--- Model has been trained ---")
    return model_pipeline


app = Flask(__name__)
model = train_model( )
UNIQUE_VALUES = {
    'regions': sorted(df['Region'].unique().tolist()),
    'neighborhoods': sorted(df['Neighborhood'].unique().tolist()),
    'property_types': sorted(df['Property_Type'].unique().tolist())
}



@app.route('/')
def home():
    return render_template('home.html' )

@app.route('/ai-estimator' )
def ai_estimator_page( ):

    return render_template('ai_estimator.html', unique_values=UNIQUE_VALUES )


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        input_df = pd.DataFrame([data])
        
        for col in ['Area_sqm', 'Bedrooms', 'Property_Age_Years']:
            input_df[col] = pd.to_numeric(input_df[col])

        
        prediction = model.predict(input_df)
        predicted_price = prediction[0]

        
        df_scored = df.copy()
        df_scored['Price_Difference'] = abs(df_scored['Price_SAR'] - predicted_price)
        
        user_region = data['Region']
        user_type = data['Property_Type']

       
        tier1 = df_scored[
            (df_scored['Region'] == user_region) & 
            (df_scored['Property_Type'] == user_type)
        ].sort_values('Price_Difference')

        
        tier2 = df_scored[
            (df_scored['Property_Type'] == user_type) & 
            (df_scored['Region'] != user_region)
        ].sort_values('Price_Difference')

        
        tier3 = df_scored[
            (df_scored['Region'] == user_region) & 
            (df_scored['Property_Type'] != user_type)
        ].sort_values('Price_Difference')

        
        recommendations_combined = pd.concat([tier1, tier2, tier3])

        
        top_recommendations = recommendations_combined.head(3).to_dict('records')

        return jsonify({
            'predicted_price': predicted_price,
            'recommendations': top_recommendations
        })
    
    except Exception as e:
        print(f"Error during prediction or recommendation: {e}")
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)