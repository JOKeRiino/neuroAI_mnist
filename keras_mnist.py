import cv2
import numpy as np
from collections import deque
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, Flatten
import tensorflow as tf
import matplotlib.pyplot as plt

# globals
model_file = 'mnist_model.keras'
buffer_size = 30
frame_buffer = deque(maxlen=buffer_size)
digit = "N/A"
recording = False
doTraining = False

# Load and preprocess MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

if not doTraining:
    # Load the pre-trained model
    model = load_model(model_file)
    print("Loaded saved model from disk.")
    _, acc = model.evaluate(x_test, y_test, verbose=0)
    print('> %.3f' % (acc * 100.0))
else:
    # Build the CNN model
    model = Sequential()
    model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(28, 28, 1)))
    model.add(Conv2D(32, kernel_size=3, activation='relu'))
    model.add(Flatten())
    model.add(Dense(10, activation='softmax'))
    model.compile(optimizer=tf.optimizers.RMSprop(0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10)
    model.save(model_file)
    model = load_model(model_file)
    _, acc = model.evaluate(x_test, y_test, verbose=0)
    print('> %.3f' % (acc * 100.0))

# Initialize webcam
cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY)
    height, width, _ = frame.shape
    square_size = min(height, width)
    x = (width - square_size) // 2
    y = (height - square_size) // 2

    cropped_frame = thresh[y:y + square_size, x:x + square_size]
    feed = frame[y:y + square_size, x:x + square_size]
    blurred_frame = cv2.GaussianBlur(cropped_frame, (5, 5), 10)
    resized_frame = cv2.resize(blurred_frame, (28, 28))
    resized_frame = cv2.bitwise_not(resized_frame)
    resized_frame = resized_frame.astype('float32') / 255

    # Store in buffer when 'r' is pressed
    if cv2.waitKey(1) & 0xFF == ord('r'):
        recording = True

    if recording:
        frame_buffer.append(resized_frame)

        if len(frame_buffer) == buffer_size:
            combined_frame = np.mean(np.array(frame_buffer), axis=0)
            combined_frame_reshaped = combined_frame.reshape(1, 28, 28, 1)

            # Predict the digit
            prediction = model.predict(combined_frame_reshaped)
            print(prediction)
            digit = np.argmax(prediction)

            combined_frame_output = cv2.resize(combined_frame_reshaped[0], (280, 280), interpolation=cv2.INTER_NEAREST)
            cv2.imshow('Combined Frame Input', combined_frame_output)

            plt.figure(figsize=(8, 4))
            plt.subplot(1, 2, 1)
            plt.imshow(combined_frame, cmap="gray")
            plt.axis("off")
            plt.subplot(1, 2, 2)
            y_positions = range(len(prediction[0]))
            plt.barh(y_positions, prediction[0], align="center")
            plt.yticks(y_positions, [str(i) for i in range(10)])
            plt.xlabel("Probability")
            plt.ylabel("Classes")
            plt.tight_layout()
            plt.show()

            frame_buffer = []
            recording = False

    cv2.putText(feed, "Digit: " + str(digit) + ("  *REC*" if recording else ""), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow('Cropped Webcam Feed', feed)
    cv2.imshow('Blurred Webcam Feed', blurred_frame)

    # Exit loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
