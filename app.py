from flask import Flask, request, render_template
import numpy as np
import pickle

# Load the model and scaler
model = pickle.load(open("regmodel.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

app = Flask(__name__)

# Route to render the home page
@app.route('/')
def index():
    return render_template("home.html")

# Route to handle predictions
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Extract input data and convert to float
        data = [float(x) for x in request.form.values()]

        # Scale the input data
        scaled_data = scaler.transform(np.array(data).reshape(1, -1))

        # Make prediction
        predicted_data = model.predict(scaled_data)[0]

        # Render the result on the page
        return render_template("home.html", prediction_text=f"Predicted House Price is ${predicted_data:.2f}")

    except ValueError:
        # Handle invalid input
        return render_template("home.html", prediction_text="Error: Please enter valid numeric values.")

if __name__ == "__main__":
    app.run(debug=True)
