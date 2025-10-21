# Add this import at the top
import json
from firebase_functions import https_fn, options
from firebase_admin import initialize_app
import numpy as np
import tensorflow.lite as tflite

initialize_app()

# Global dictionaries to hold models and paths
models = {}
model_paths = {
    "course_career": "model_nn_course_career.tflite",
    "course_job": "model_nn_course_job.tflite",
    "time_series": "model_nn_time_series.tflite"
}

@https_fn.on_request(
    memory=options.MemoryOption.GB_1,

    # --- THIS IS THE FIX FOR THE CORS ERROR ---
    cors=options.CorsOptions(
        cors_origins=["*"],  # Allows all domains (like localhost)
        cors_methods=["post"] # Allows POST requests
    )
    # --- END OF FIX ---

)
def predict(req: https_fn.Request) -> https_fn.Response:
    """
    An HTTPS endpoint that runs predictions. Models are loaded into memory
    on the first request that needs them.
    """
    try:
        request_data = req.get_json(silent=True)
        if not request_data:
            # FIX 1: Use json.dumps
            return https_fn.Response('{"error": "Invalid JSON."}', status=400, mimetype="application/json")

        model_name = request_data.get('model_name')
        input_data = request_data.get('inputs')

        if not model_name or model_name not in model_paths:
            # FIX 2: Use json.dumps
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
        
        # FIX 3: Use json.dumps
        return https_fn.Response(json.dumps(response_body), status=200, mimetype="application/json")

    except Exception as e:
        print(f"An internal error occurred: {e}")
        error_payload = {
            "error": f"An internal error occurred: {str(e)}"
        }
        # FIX 4: Use json.dumps
        return https_fn.Response(json.dumps(error_payload), status=500, mimetype="application/json")