import pickle
from flask import Flask, jsonify, request

# List of penguin species classes
classes = ['Adelie', 'Chinstrap', 'Gentoo']

def predict_single(penguin, dv, model):
    # Prepare the penguin data for prediction
    penguin_dict = {
        'island': penguin['island'],
        'sex': penguin['sex'],
        'bill_length_mm': penguin['bill_length_mm'],
        'bill_depth_mm': penguin['bill_depth_mm'],
        'flipper_length_mm': penguin['flipper_length_mm'],
        'body_mass_g': penguin['body_mass_g']
    }
    
    penguin_transformed = dv.transform([penguin_dict])
    y_pred = model.predict(penguin_transformed)[0]
    y_prob = model.predict_proba(penguin_transformed)[0][y_pred]
    
    return (y_pred, y_prob)

def predict(dv, model):
    # Get the penguin data from the request body (JSON)
    penguin = request.get_json()
    
    species, probability = predict_single(penguin, dv, model)
    
    result = {
        'species': classes[species],
        'probability': float(probability)
    }
    return jsonify(result)

# Initialize Flask app
app = Flask('penguin_species')

# Define routes for each model
@app.route('/predict_lr', methods=['POST'])
def predict_lr():
    with open('models/species-model.pck', 'rb') as f:
        dv, model = pickle.load(f)
    return predict(dv, model)

@app.route('/predict_svm', methods=['POST'])
def predict_svm():
    with open('models/species-svm.pck', 'rb') as f:
        dv, model = pickle.load(f)
    return predict(dv, model)

@app.route('/predict_dt', methods=['POST'])
def predict_dt():
    with open('models/species-dt.pck', 'rb') as f:
        dv, model = pickle.load(f)
    return predict(dv, model)

@app.route('/predict_knn', methods=['POST'])
def predict_knn():
    with open('models/species-knn.pck', 'rb') as f:
        dv, model = pickle.load(f)
    return predict(dv, model)

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True, port=8000)
