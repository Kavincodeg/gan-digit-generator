import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt

# Load dataset
(x_train, _), (_, _) = tf.keras.datasets.mnist.load_data()

# Normalize
x_train = x_train / 127.5 - 1.0
x_train = np.expand_dims(x_train, axis=-1)

BATCH_SIZE = 128
dataset = tf.data.Dataset.from_tensor_slices(x_train).shuffle(60000).batch(BATCH_SIZE)

# Generator
def build_generator():
    model = tf.keras.Sequential([
        layers.Dense(128, activation='relu', input_shape=(100,)),
        layers.Dense(256, activation='relu'),
        layers.Dense(28*28, activation='tanh'),
        layers.Reshape((28, 28, 1))
    ])
    return model

# Discriminator
def build_discriminator():
    model = tf.keras.Sequential([
        layers.Flatten(input_shape=(28, 28, 1)),
        layers.Dense(256, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    return model

generator = build_generator()
discriminator = build_discriminator()

discriminator.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

discriminator.trainable = False

gan_input = tf.keras.Input(shape=(100,))
fake_image = generator(gan_input)
gan_output = discriminator(fake_image)

gan = tf.keras.Model(gan_input, gan_output)
gan.compile(optimizer='adam', loss='binary_crossentropy')

EPOCHS = 20
noise_dim = 100

def train():
    for epoch in range(EPOCHS):
        for real_images in dataset:
            batch_size = real_images.shape[0]

            noise = np.random.normal(0, 1, (batch_size, noise_dim))
            fake_images = generator.predict(noise, verbose=0)

            real_labels = np.ones((batch_size, 1))
            fake_labels = np.zeros((batch_size, 1))

            discriminator.trainable = True
            discriminator.train_on_batch(real_images, real_labels)
            discriminator.train_on_batch(fake_images, fake_labels)

            noise = np.random.normal(0, 1, (batch_size, noise_dim))
            misleading_labels = np.ones((batch_size, 1))

            discriminator.trainable = False
            gan.train_on_batch(noise, misleading_labels)

        print(f"Epoch {epoch+1} completed")

    generate_images()

def generate_images():
    noise = np.random.normal(0, 1, (16, noise_dim))
    generated_images = generator.predict(noise)

    generated_images = 0.5 * generated_images + 0.5

    plt.figure(figsize=(4,4))
    for i in range(16):
        plt.subplot(4,4,i+1)
        plt.imshow(generated_images[i, :, :, 0], cmap='gray')
        plt.axis('off')
    plt.show()

train()