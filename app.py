import joblib
import numpy as np
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)


@app.route("/home", methods=["GET"])
def home():
    return render_template("home.html")


@app.route("/asjson", methods=["GET"])
def get_json_data():
    return jsonify({"response": "Data from the Flask is in json format"})


@app.route("/predict", methods=["POST"])
def predict_iris():

    request_body = request.get_json(force=True)

    petal_length = float(request_body["petal_length"])
    petal_width = float(request_body["petal_width"])
    sepal_length = float(request_body["sepal_length"])
    sepal_width = float(request_body["sepal_width"])

    model = joblib.load("iris_classifier.pkl")

    test_data = np.array(
        [petal_length, petal_width, sepal_length, sepal_width]
    ).reshape(1, 4)

    class_prediced = int(model.predict(test_data)[0])

    class_type = ["setosa", "versicolor", "virginica"]

    output = "Predicted Iris Class: " + class_type[class_prediced]

    return jsonify({"response": output})


if __name__ == "__main__":
    app.run(debug=True)
