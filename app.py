from flask import Flask, render_template, request

import pandas as pd
import joblib
import os



app = Flask(__name__)

model_file = 'model.pkl'
absolute_model_path = os.path.abspath(model_file)
rf_model = joblib.load(open(absolute_model_path, 'rb'))


def predict_oral_cancer_cl(data):
    try:
        if data.shape[1] != len(rf_model.feature_importances_):
            raise ValueError("Input data shape does not match model's feature count.")
        
        predictions = rf_model.predict(data)
        return round(predictions[0], 3)
    except Exception as e:
        print(f"Error during prediction: {e}")
        return None
    

def get_hospitals_by_district(district, max_hospitals=5):
    try:
        # Load the CSV file into a DataFrame
        df = pd.read_csv('hospi.csv')

        # Filter hospitals by district
        filtered_hospitals = df[df['District'] == district]

        # Get up to max_hospitals from the filtered DataFrame
        hospitals_list = []
        for index, row in filtered_hospitals.iterrows():
            hospital_details = {
                'Name': row['Name'],
                'Contact': row['Contact'],
                'Address': row['Address']
                # Add more details if needed
            }
            hospitals_list.append(hospital_details)

            # Break the loop if reached maximum hospitals
            if len(hospitals_list) >= max_hospitals:
                break

        return hospitals_list
    except Exception as e:
        print(f"Error loading hospitals: {e}")
        return []

   
@app.route('/', methods=['GET', 'POST'])
def home():
    
    if request.method == 'POST':
        input_data = {
            'localization': int(request.form['localization']),
            'size': request.form['size'],
            'tobacco_use': int(request.form['tobacco_use']),
            'alcohol_consumption': int(request.form['alcohol_consumption']),
            'sun_exposure': int(request.form['sun_exposure']),
            'gender': int(request.form['gender']),
            'age_group': int(request.form['age_group'])
        }

        input_df = pd.DataFrame([input_data])

        district = request.form['district']
        hospitals_list = get_hospitals_by_district(district)
        result_cl = predict_oral_cancer_cl(input_df)

        return render_template('result.php', result_cl=result_cl, hospitals_list=hospitals_list)

    return render_template('index.php')
@app.route('/causes')
def causes():
    return render_template('causes.html')
@app.route('/about_oc')
def about_oc():
    return render_template('about_oc.html')
@app.route('/treatment')
def treatment():
    return render_template('treatment.html')
@app.route('/symptoms')
def symptoms():
    return render_template('symptoms.html')

   


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=10000)  # Specify the port (e.g., 8000) and host
