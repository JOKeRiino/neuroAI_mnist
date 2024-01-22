import subprocess


def main(name):
    cnn = False  # Set this to "False" to run the Nengo SNN

    # If you want to train the model, set doTraining=True in the corresponding file.

    if cnn:
        subprocess.call(['python', 'keras_mnist.py'])
    else:
        subprocess.call(['python', 'nengo_mnist.py'])


if __name__ == '__main__':
    main('PyCharm')
