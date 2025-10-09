# Add this import at the top
import json 
from firebase_functions import https_fn, options
from firebase_admin import initialize_app
import numpy as np
import tensorflow.lite as tflite
from flask import Flask, request, jsonify

initialize_app()

# Global dictionaries to hold models and paths
models = {}
model_paths = {
    "course_career": "model_nn_course_career.tflite",
    "course_job": "model_nn_course_job.tflite",
    "time_series": "model_nn_time_series.tflite"
}

@https_fn.on_request(memory=options.MemoryOption.GB_1)
def predict(req: https_fn.Request) -> https_fn.Response:
    """
    An HTTPS endpoint that runs predictions. Models are loaded into memory
    on the first request that needs them.
    """
    try:
        request_data = req.get_json(silent=True)
        if not request_data:
            return https_fn.Response('{"error": "Invalid JSON."}', status=400, mimetype="application/json")

        model_name = request_data.get('model_name')
        input_data = request_data.get('inputs')

        if not model_name or model_name not in model_paths:
            return https_fn.Response('{"error": "Invalid model name."}', status=400, mimetype="application/json")

        # Lazy-loading logic: only load the model if it's not in memory
        if model_name not in models:
            print(f"Loading model '{model_name}' for the first time...")
            interpreter = tflite.Interpreter(model_path=model_paths[model_name])
            interpreter.allocate_tensors()
            models[model_name] = interpreter
            print(f"Model '{model_name}' loaded successfully.")

        interpreter = models[model_name]

        # Prediction logic
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        input_tensor = np.array(input_data, dtype=np.float32)

        if len(input_tensor.shape) == len(input_details[0]['shape']) - 1:
            input_tensor = np.expand_dims(input_tensor, axis=0)

        interpreter.set_tensor(input_details[0]['index'], input_tensor)
        interpreter.invoke()

        output_data = interpreter.get_tensor(output_details[0]['index'])
        prediction = output_data[0].tolist()
        
        response_body = {
            "prediction_for": model_name,
            "prediction": prediction
        }
        
        # --- THIS IS THE CORRECTED LINE ---
        # Convert the dictionary to a JSON string and set the correct mimetype
        return https_fn.Response(json.dumps(response_body), status=200, mimetype="application/json")

    except Exception as e:
        print(f"An internal error occurred: {e}")
        error_payload = {
            "error": f"An internal error occurred: {str(e)}"
        }
        return https_fn.Response(json.dumps(error_payload), status=500, mimetype="application/json")
        # -----------------------------------------------------------
# THIS IS THE PART YOU NEED TO FIND AND FIX AT THE BOTTOM
# -----------------------------------------------------------

# Move the app creation INSIDE the if block as well.
if __name__ == "__main__":
    app = Flask(__name__)
    # If you have any @app.route decorators for local testing,
    # they would need to be defined in here too.
    app.run(host='0.0.0.0', port=8081)