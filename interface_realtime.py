import cv2
import numpy as np
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model('mnist_cnn_model.h5')

# Function to preprocess frame for digit recognition
def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    resized = cv2.resize(gray, (28, 28))            # Resize to 28x28
    inverted = cv2.bitwise_not(resized)             # Invert colors (MNIST has white digits on black background)
    normalized = inverted / 255.0                   # Normalize pixel values
    reshaped = normalized.reshape(1, 28, 28, 1)     # Reshape for the model input
    return reshaped

# Function to predict the digit from a frame
def predict_digit(frame):
    preprocessed_frame = preprocess_frame(frame)
    prediction = model.predict(preprocessed_frame)
    digit = np.argmax(prediction)
    return digit

# Initialize webcam
cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    print("Error: Could not open video capture.")
    exit()

# Set camera to fullscreen
cv2.namedWindow('Digit Recognizer', cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty('Digit Recognizer', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        print("Error: Unable to capture video frame.")
        break

    # Get frame dimensions
    height, width, _ = frame.shape

    # Define a slightly smaller centered region of interest (ROI) for digit recognition
    square_size = 250  # Reduced square size to 250x250
    top_left_x = (width - square_size) // 2
    top_left_y = (height - square_size) // 2
    bottom_right_x = top_left_x + square_size
    bottom_right_y = top_left_y + square_size

    roi = frame[top_left_y:bottom_right_y, top_left_x:bottom_right_x]  # The centered square region

    # Predict the digit in the ROI
    digit = predict_digit(roi)

    # Display the prediction on the frame
    cv2.putText(frame, f"Predicted Digit: {digit}", (50, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Draw a rectangle around the ROI
    cv2.rectangle(frame, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), (255, 0, 0), 2)

    # Show the frame with the prediction
    cv2.imshow('Digit Recognizer', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
