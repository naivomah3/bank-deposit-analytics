from flask import Flask, request, make_response, jsonify
import pickle
import joblib
import flasgger
from helper import error_msg, ord_bin_enc, ohe_encoding, yeoj_transform
import pandas as pd


app = Flask(__name__)
flasgger.Swagger(app)
# Load components: model and encoders
model = joblib.load("objects/clf_xgb_final.model")
ohe_enc = pickle.load(open("objects/ohe_encoder.obj", "rb"))
rare_enc = pickle.load(open("objects/rare_encoder.obj", "rb"))
woe_enc = pickle.load(open("objects/woe_encoder.obj", "rb"))
yeoj_trans = pickle.load(open("objects/yeoj_transformer.obj", "rb"))


@app.route('/')
def home():
    return "Server is running, you can now make a JSON request."


@app.route('/predict', methods=['GET'])
def predict_deposit():
    if not request.is_json:
        return "Request is not JSON, you can now make a JSON request.", 400
    json_data = request.json

    age = json_data["age"]
    if not isinstance(age, int):
        return error_msg(age)

    job = json_data["job"]
    if job not in ['blue-collar', 'management', 'technician',
                   'admin.', 'services', 'retired',
                   'self-employed', 'entrepreneur',
                   'unemployed', 'housemaid', 'student',
                   'unknown']:
        return error_msg(job)

    marital = json_data["marital"]
    if marital not in ['married', 'single', 'divorced']:
        return error_msg(marital)

    education = json_data["education"]
    if education not in ['secondary', 'tertiary', 'primary', 'unknown']:
        return error_msg(education)

    default = json_data["default"]
    if default not in ['no', 'yes']:
        return error_msg(default)

    balance = json_data["balance"]
    if not isinstance(balance, int):
        return error_msg(balance)

    housing = json_data["housing"]
    if housing not in ['no', 'yes']:
        return error_msg(housing)

    loan = json_data["loan"]
    if loan not in ['no', 'yes']:
        return error_msg(loan)

    contact = json_data["contact"]
    if contact not in ['cellular', 'unknown', 'telephone']:
        return error_msg(contact)

    day = json_data["day"]
    if not isinstance(day, int):
        return error_msg(day)

    month = json_data["month"]
    if month not in ['may', 'jul', 'aug', 'jun', 'nov', 'apr', 'feb', 'jan', 'oct', 'sep', 'mar', 'dec']:
        return error_msg(month)

    duration = json_data["duration"]
    if not isinstance(duration, int):
        return error_msg(duration)

    campaign = json_data["campaign"]
    if not isinstance(campaign, int):
        return error_msg(campaign)

    pdays = json_data["pdays"]
    if not isinstance(pdays, int):
        return error_msg(pdays)

    previous = json_data["previous"]
    if not isinstance(previous, int):
        return error_msg(previous)

    poutcome = json_data["poutcome"]
    if poutcome not in ['unknown', 'failure', 'other', 'success']:
        return error_msg(poutcome)

    # JSON to Dataframe
    data = pd.json_normalize(json_data)
    # Ordinal Encode
    data_enc = ord_bin_enc(data)
    # Rare Encode
    data_enc = rare_enc.transform(data_enc)
    # WoE Encode
    data_enc = woe_enc.transform(data_enc)
    # OHE Encode
    data_enc = ohe_encoding(data_enc, ohe_enc)
    # Yeo-J tranformation
    data_enc = yeoj_transform(data_enc, yeoj_trans)
    # Do predict
    pred = model.predict(data_enc)

    subscribe = "No"
    if pred == 1:
        subscribe = "Yes"

    response_body = {
        "subscription": subscribe,
    }

    return make_response(jsonify(response_body), 200)


if __name__ == "__main__":
    app.run(debug=True)
