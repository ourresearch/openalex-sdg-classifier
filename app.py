from flask import Flask, request, jsonify
from helpers import get_predictions

app = Flask(__name__)


@app.route('/classify/', methods=['POST'])
def classify():
    # use the param 'text' to get the text to classify
    text = request.json.get('text', None)
    if not text:
        return jsonify({"msg": "Missing 'text' in request data"}), 400

    try:
        result = get_predictions(text)
        return jsonify(result)
    except Exception as e:
        return jsonify({"msg": str(e)}), 500


if __name__ == "__main__":
    app.run()
