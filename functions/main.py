# Add this import at the top
import json
from firebase_functions import https_fn, options
from firebase_admin import initialize_app
import numpy as np
import tensorflow.lite as tflite

# Initialize Firebase Admin SDK
initialize_app()

# Global dictionaries to hold loaded models and their file paths
models = {}
model_paths = {
    "course_career": "model_nn_course_career.tflite",
    "course_job": "model_nn_course_job.tflite",
    "time_series": "model_nn_time_series.tflite"
}

@https_fn.on_request(
    memory=options.MemoryOption.GB_1, # Set memory for the function

    # --- CORS FIX ---
    # Allow requests from web apps hosted on different domains (like localhost)
    cors=options.CorsOptions(
        cors_origins=["*"],  # Allows all domains
        cors_methods=["post"] # Allows POST requests
    )
    # --- END OF CORS FIX ---
)
def predict(req: https_fn.Request) -> https_fn.Response:
    """
    An HTTPS endpoint that runs predictions using TFLite models.
    Models are loaded into memory on the first request that needs them.
    """
    try:
        # --- Request Handling ---
        # Get JSON data from the request body
        request_data = req.get_json(silent=True)
        if not request_data:
            # Return error if JSON is invalid or missing
            return https_fn.Response('{"error": "Invalid JSON."}', status=400, mimetype="application/json")

        # Extract model name and input data from JSON
        model_name = request_data.get('model_name')
        input_data = request_data.get('inputs')

        # Validate the model name
        if not model_name or model_name not in model_paths:
            # Return error if model name is invalid
            return https_fn.Response('{"error": "Invalid model name."}', status=400, mimetype="application/json")
        
        # Validate input data presence
        if input_data is None:
             return https_fn.Response('{"error": "Missing input data."}', status=400, mimetype="application/json")


        # --- Model Loading (Lazy Loading) ---
        # Load the model only if it hasn't been loaded before
        if model_name not in models:
            print(f"Loading model '{model_name}' for the first time...")
            # Load the TFLite model file
            interpreter = tflite.Interpreter(model_path=model_paths[model_name])
            # Allocate memory for the model's tensors
            interpreter.allocate_tensors()
            # Store the loaded interpreter in the global dictionary
            models[model_name] = interpreter
            print(f"Model '{model_name}' loaded successfully.")

        # Get the interpreter for the requested model
        interpreter = models[model_name]

        # --- Prediction Logic ---
        # Get details about the model's input and output tensors
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        # Convert the input data into a NumPy array with the correct data type
        # This handles lists, nested lists etc.
        try:
             input_tensor = np.array(input_data, dtype=np.float32)
        except Exception as conversion_error:
             print(f"Error converting input data to NumPy array: {conversion_error}")
             return https_fn.Response(json.dumps({"error": f"Invalid input data format: {conversion_error}"}), status=400, mimetype="application/json")


        # --- Shape Handling & Debugging ---
        model_expected_shape = input_details[0]['shape']
        print(f"--- DEBUGGING ---")
        print(f"Model expected shape: {model_expected_shape}")
        print(f"Input tensor shape BEFORE reshape attempt: {input_tensor.shape}")

        # Try to reshape the input tensor to match the model's expected shape
        # This handles cases like adding batch dimension or matching multi-dimensional shapes
        try:
            # Check if shapes already match (ignoring potential batch dim placeholder like -1 or None)
            # We compare ranks and non-batch dimensions
            expected_rank = len(model_expected_shape)
            actual_rank = len(input_tensor.shape)
            
            # Simple case: add batch dimension if needed (e.g., input is (15,) model expects (1, 15))
            if actual_rank == expected_rank - 1 and model_expected_shape[0] == 1:
                 input_tensor = np.expand_dims(input_tensor, axis=0)
                 print(f"DEBUG: Input tensor shape AFTER adding batch dim: {input_tensor.shape}")
            # More general reshape if shapes don't match after potential batch dim add
            elif tuple(input_tensor.shape) != tuple(model_expected_shape):
                 # Attempt a direct reshape, this might fail if dimensions don't multiply correctly
                 input_tensor = input_tensor.reshape(model_expected_shape)
                 print(f"DEBUG: Input tensor shape AFTER direct reshape: {input_tensor.shape}")

        except ValueError as reshape_error:
             print(f"Error reshaping input tensor: {reshape_error}. Input shape {input_tensor.shape} cannot be reshaped to {model_expected_shape}.")
             return https_fn.Response(json.dumps({"error": f"Input data shape mismatch. Expected compatible with {model_expected_shape}, but got {input_tensor.shape}. Reshape error: {reshape_error}"}), status=400, mimetype="application/json")

        # Final shape check after potential reshaping
        if tuple(input_tensor.shape) != tuple(model_expected_shape):
            final_error_msg = f"Input tensor final shape {input_tensor.shape} does not match model expected shape {model_expected_shape} after attempts to reshape."
            print(final_error_msg)
            return https_fn.Response(json.dumps({"error": final_error_msg}), status=400, mimetype="application/json")


        # --- Run Inference ---
        # Set the input tensor data into the model's interpreter
        interpreter.set_tensor(input_details[0]['index'], input_tensor)
        # Run the model inference
        interpreter.invoke()

        # Get the prediction result from the output tensor
        output_data = interpreter.get_tensor(output_details[0]['index'])
        # Convert the output (often a NumPy array) to a standard Python list
        # We take the first element [0] assuming a batch size of 1 was used/expected
        prediction = output_data[0].tolist() if output_data.shape[0] == 1 else output_data.tolist()

        # --- Prepare Response ---
        # Create the response body as a Python dictionary
        response_body = {
            "prediction_for": model_name,
            "prediction": prediction
        }

        # --- JSON RESPONSE FIX ---
        # Convert the dictionary to a JSON string and return it with status 200 (OK)
        return https_fn.Response(json.dumps(response_body), status=200, mimetype="application/json")

    except Exception as e:
        # --- Error Handling ---
        # Log the detailed error to Firebase console
        print(f"An internal error occurred during prediction: {e}")
        # Create a JSON error response
        error_payload = {
            "error": f"An internal server error occurred: {str(e)}"
        }
        # Convert error payload to JSON and return with status 500 (Internal Server Error)
        return https_fn.Response(json.dumps(error_payload), status=500, mimetype="application/json")


# --- DEPLOYMENT TIMEOUT FIX ---
# This block is intentionally left minimal for deployment.
# Local testing code (like a Flask server) should NOT be placed here.
if __name__ == "__main__":
    print("main.py executed directly (likely for local testing, not deployment).")
    # To run local tests, you would typically use the Firebase Emulator Suite
    # or call the functions/methods directly in a separate test script.