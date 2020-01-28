from flask import Flask, escape, request
from flask import jsonify
from toxic_comments.online_predictor.predictor import ToxicClassifier

app = Flask(__name__)
model = ToxicClassifier()

@app.route('/predict', methods=['POST'])
def get_prediction():
    try: 
        payload=request.json['examples']
        examples = [e['comment'] for e in payload]
        predictions = model.predict(examples)
    except Exception as e:
        return jsonify(error=f"Something is wrong with input.\nExpected JSON of the following structure: {{'examples':[{{'comment': 'some text'}}]}}, got {request.json}.\nError: {e}")
    return jsonify(predictions=[p for p in predictions])
    
if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0', port=8070)