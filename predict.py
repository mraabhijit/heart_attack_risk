import pickle
from flask import Flask, request, jsonify

model_file = 'model.bin'

with open(model_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)

app = Flask('risk')

def get_risk(patient):

    X = dv.transform([patient])
    y_pred = model.predict_proba(X)[0, 1]
    risk = y_pred >= 0.5

    return y_pred, risk


@app.route('/predict', methods=['POST'])
def predict_prob():
    patient = request.get_json()

    y_pred, risk = get_risk(patient)

    result = {
        'risk_probability': float(round(y_pred, 3)),
        'attack risk': bool(risk)
    }   

    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8080)