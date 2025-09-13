"""
yolo_functions.py
-----------------
This module provides utility functions for loading YOLO models, performing inference on images or videos, and displaying annotated results. It is designed to support rugby analytics and other computer vision tasks using the Ultralytics YOLO framework.

Functions:
---------------
- `load_model`: Loads a YOLO model from the specified file path.
- `perform_inference`: Runs inference on an image or video file using the provided YOLO model and returns a generator of results.
- `show_annotated_file`: Displays annotated frames or images from YOLO inference results, with interactive pause and resume controls.

Dependencies:
---------------
- OpenCV (`cv2`)
- ultralytics (`YOLO`)
- general_functions
"""

import cv2, sys
from ultralytics import YOLO

def load_model(model_path):
    """
    Loads a YOLO model from the specified file path.

    Parameters
        model_path (str): The file path for the YOLO model.

    Returns:
        YOLO: The initialised YOLO model object.

    Raises:
        SystemExit: If the model cannot be loaded due to an error.
    """
    try:
        return YOLO(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)


def perform_inference(file, model, show_output=True, conf=0.25):
    """
    Runs inference on an image or video using a YOLO model and returns a generator of results.

    Parameters
        file (str or np.ndarray): The file path to the image or video to process, or the image itself as a NumPy array.
        model (YOLO): The YOLO model used for inference.
        show_output (bool, optional): Whether to output results to the console. Defaults to True.
        conf (float, optional): Confidence threshold for prediction. Defaults to 0.25.

    Returns:
        generator: A generator yielding results for the image or each frame in the video.

    Raises:
        SystemError: If an error occurs during inference.
    """
    try:
        return model.predict(file, stream=True, verbose=show_output, conf=conf)
    except Exception:
        raise SystemError("Error when running inference")


def show_annotated_file(inference_results):
    """
    Displays video frames or images with annotations based on YOLO inference results. Supports interactive pausing and resuming.

    Parameters
        inference_results (generator): A generator yielding results for each frame in the video or image.

    Returns:
        None
    """
    paused = False

    for result in inference_results:

        if paused:  # If the video is paused, wait for user input to resume or exit
            key = cv2.waitKey(0)  # Wait indefinitely for a key press
            if key & 0xFF == ord('r'):  # Resume when 'r' key is pressed
                paused = False
            elif key & 0xFF == ord('q'):  # Exit when 'q' key is pressed
                break

        annotated_frame = result.plot()
        cv2.imshow("Model Inference", annotated_frame)

        # Wait for key press to pause the video
        key = cv2.waitKey(1)  # Process key press for pausing
        if key & 0xFF == ord('s'):  # Pause when 's' key is pressed
            paused = True

        # Exit when 'q' key is pressed
        if key & 0xFF == ord('q'):
            break
    
    print("Stream finished, press 'q' to quit")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
