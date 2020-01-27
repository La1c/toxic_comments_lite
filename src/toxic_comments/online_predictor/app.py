from flask import Flask, escape, request
from flask import jsonify
from toxic_comments.online_predictor.predictor import ToxicClassifier

app = Flask(__name__)
model = ToxicClassifier()

@app.route('/predict', methods=['POST'])
def get_prediction():
    payload=request.json['examples']
    examples = [e['comment'] for e in payload]
    predictions = model.predict(examples)
    return jsonify(predictions=[p for p in predictions])
    
if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0', port=8070)