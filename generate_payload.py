import json

# Wrap all the script's logic in this block
if __name__ == "__main__":
    # The data your model needs
    payload_data = {
      "model_name": "course_job",
      "inputs": [0.0] * 1031  # CORRECTED SIZE
    }

    # Write the data to a file named payload.json
    with open('payload.json', 'w') as f:
        json.dump(payload_data, f)

    print("Successfully created 'payload.json' with the correct size.")