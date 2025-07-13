from flask import Flask, request, jsonify, render_template
import os
from flask_cors import CORS, cross_origin
from cnnClassifier.utils.common import decodeImage
from cnnClassifier.pipeline.prediction import PredictionPipeline
from cnnClassifier import logger

# Set environment variables (locale settings for compatibility)
os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS

# App-level classifier wrapper
class ClientApp:
    def __init__(self):
        self.filename = "inputImage.jpg"
        self.classifier = PredictionPipeline(self.filename)

# Create an instance of ClientApp globally
clApp = ClientApp()

# Route: Home page
@app.route("/", methods=['GET'])
@cross_origin()
def home():
    return render_template('index.html')

# Route: Optional training trigger
@app.route("/train", methods=['GET', 'POST'])
@cross_origin()
def trainRoute():
    try:
        os.system("python main.py")
        return "Training done successfully!"
    except Exception as e:
        logger.error(f"Training failed: {e}")
        return jsonify({"error": str(e)})

# Route: Prediction
@app.route("/predict", methods=['POST'])
@cross_origin()
def predictRoute():
    try:
        image = request.json['image']  # base64 image
        decodeImage(image, clApp.filename)  # Save to disk
        result = clApp.classifier.predict()  # Call prediction pipeline
        return jsonify(result)  # Return prediction result
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        return jsonify({"error": "Prediction failed", "details": str(e)})

# Entry point
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080, debug=True)
