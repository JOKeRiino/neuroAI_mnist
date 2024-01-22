import nengo
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import nengo_dl
import cv2
from collections import deque

# globals
minibatch_size = 1
n_steps = 30
buffer_size = 30
frame_buffer = deque(maxlen=buffer_size)
digit = "N/A"
recording = False
do_training = False  # Before training minibatch_size must be set to around 300

(train_images, train_labels), (
    test_images,
    test_labels,
) = tf.keras.datasets.mnist.load_data()

train_images = train_images.reshape((train_images.shape[0], -1))
test_images = test_images.reshape((test_images.shape[0], -1))
train_images = train_images[:, None, :]
train_labels = train_labels[:, None, None]
test_images = np.tile(test_images[:, None, :], (1, n_steps, 1))
test_labels = np.tile(test_labels[:, None, None], (1, n_steps, 1))


def classification_accuracy(y_true, y_pred):
    return tf.metrics.sparse_categorical_accuracy(y_true[:, -1], y_pred[:, -1])


with nengo.Network(seed=0) as net:
    # set some default parameters for the neurons
    net.config[nengo.Ensemble].max_rates = nengo.dists.Choice([100])
    net.config[nengo.Ensemble].intercepts = nengo.dists.Choice([0])
    net.config[nengo.Connection].synapse = None
    neuron_type = nengo.LIF(amplitude=0.01)

    nengo_dl.configure_settings(stateful=False)
    inp_layer = nengo.Node(np.zeros(28 * 28))

    layer = nengo_dl.Layer(tf.keras.layers.Conv2D(filters=32, kernel_size=3))(
        inp_layer, shape_in=(28, 28, 1)
    )
    layer = nengo_dl.Layer(neuron_type)(layer)

    layer = nengo_dl.Layer(tf.keras.layers.Conv2D(filters=64, strides=2, kernel_size=3))(
        layer, shape_in=(26, 26, 32)
    )
    layer = nengo_dl.Layer(neuron_type)(layer)

    layer = nengo_dl.Layer(tf.keras.layers.Conv2D(filters=128, strides=2, kernel_size=3))(
        layer, shape_in=(12, 12, 64)
    )
    layer = nengo_dl.Layer(neuron_type)(layer)

    out_layer = nengo_dl.Layer(tf.keras.layers.Dense(units=10))(layer)
    out_p = nengo.Probe(out_layer, label="out_p")
    out_p_filt = nengo.Probe(out_layer, synapse=0.1, label="out_p_filt")

sim = nengo_dl.Simulator(net, minibatch_size=minibatch_size)

if do_training:
    # run training
    sim.compile(
        optimizer=tf.optimizers.RMSprop(0.001),
        loss={out_p: tf.losses.SparseCategoricalCrossentropy(from_logits=True)},
    )
    sim.fit(train_images, {out_p: train_labels}, epochs=10)

    sim.save_params("./mnist_params")

    sim.compile(loss={out_p_filt: classification_accuracy})
    print(
        "Accuracy after training:",
        sim.evaluate(test_images, {out_p_filt: test_labels}, verbose=0)["loss"],
    )
else:
    print("Not training, loading params from file...")
    sim.load_params("./mnist_params")

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
            combined_frame_flattened = combined_frame.reshape(-1)
            combined_frame_flattened_and_repeated = np.tile(combined_frame_flattened[None, None, :], (1, n_steps, 1))

            # Predict & plot the digit
            prediction = sim.predict(combined_frame_flattened_and_repeated)
            plt.figure(figsize=(8, 4))
            plt.subplot(1, 2, 1)
            plt.imshow(combined_frame, cmap="gray")
            plt.axis("off")

            plt.subplot(1, 2, 2)
            plt.plot(tf.nn.softmax(prediction[out_p_filt][0]))
            plt.legend([str(i) for i in range(10)], loc="upper left")
            plt.xlabel("timesteps")
            plt.ylabel("probability")
            plt.tight_layout()
            plt.show()
            digit = np.argmax(tf.nn.softmax(prediction[out_p_filt][0])[-1])

            combined_frame_output = cv2.resize(combined_frame, (280, 280), interpolation=cv2.INTER_NEAREST)
            cv2.imshow('Combined Frame Input', combined_frame_output)

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
sim.close()