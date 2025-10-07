import numpy as np
import json

# 1. The sample input you received for the 'course_job' model
sample_input = {
    'onehot_course_title_Sample Course A': 1.0, 
    'onehotcourse_organization_Sample Org X': 1.0, 
    'onehotcourse_Certificate_type_COURSE': 1.0, 
    'onehotcourse_difficulty_Beginner': 1.0, 
    'remaindercourse_rating': 4.5, 
    'remainder_course_students_enrolled': 10000.0
}

# 2. IMPORTANT: You must get the complete list of all 1031 feature names
#    in the correct order from the person who trained the model.
#    This is just a small EXAMPLE, you must replace it with your real list.
full_feature_list = [
    'remaindercourse_rating',
    'remainder_course_students_enrolled',
    'onehot_course_title_Sample Course A',
    'onehot_course_title_Sample Course B',
    'onehotcourse_organization_Sample Org X',
    'onehotcourse_Certificate_type_COURSE',
    'onehotcourse_difficulty_Beginner',
    # ... and so on for all 1031 features in the correct order
]

# Check if the list has the correct length
if len(full_feature_list) != 1031:
    print(f"Error: The feature list has {len(full_feature_list)} items, but it should have 1031.")
else:
    # 3. Create a vector of 1031 zeros
    final_input_vector = np.zeros(1031, dtype=np.float32)

    # 4. Fill in the values from your sample input
    for feature_name, value in sample_input.items():
        try:
            # Find the position (index) of the feature in the full list
            index = full_feature_list.index(feature_name)
            # Set the value at that specific position
            final_input_vector[index] = value
        except ValueError:
            print(f"Warning: Feature '{feature_name}' from your sample was not found in the full feature list.")

    # 5. Print the final result, which you can use to test your function
    print("--- Your final list of 1031 numbers is: ---")
    print(final_input_vector.tolist())