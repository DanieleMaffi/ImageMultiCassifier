import numpy as np
import matplotlib.pyplot as plt
from keras import datasets, layers, models
import cv2
import tensorflow as tf

# Unpacking two tuples from the dataset
(training_data, training_labels), (testing_data, testing_labels) = datasets.cifar10.load_data()
train, test = datasets.cifar10.load_data()
training_data, testing_data = training_data/255, testing_data/255   # Scaling the images

class_names = ['Plane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

# PLotting 16 images
for i in range(16):
    plt.subplot(4, 4, i+1)                                 # Making a sublot in a 4x4 grid for each image
    plt.xticks([])                                         # Removing axis's numebrs
    plt.yticks([])                                         # Removing axis's numebrs
    plt.imshow(training_data[i])       # Show the image
    plt.xlabel(class_names[training_labels[i][0]])         # PLtting the labels

plt.show()

training_data = training_data[:20000]
training_labels = training_labels[:20000]
testing_data = testing_data[:4000]
testing_labels = testing_labels[:4000]

'''model = models.Sequential([
    layers.Conv2D(36, (3,3), activation='relu', input_shape=(32,32,3)),
    layers.MaxPooling2D(),
    
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(),

    layers.Conv2D(36, (3,3), activation='relu'),
    layers.MaxPooling2D(),

    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')  # 10 possible outputs normalized by the softmax function (converts a vector of real numbers into a probability distribution).
])

model.compile('adam', metrics='accuracy', loss='sparse_categorical_crossentropy')

model.fit(training_data, training_labels, epochs=10, validation_data=(testing_data, testing_labels))

loss, accuracy = model.evaluate(testing_data, testing_labels)
print(f"Accuracy: {accuracy}, Loss: {loss}")

model.save("model.mdl")'''

model = models.load_model('model.mdl')

img = cv2.imread('cartest.jpg')
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()

resized_image = tf.image.resize(img, (32, 32))
resize_and_colored = cv2.cvtColor(resized_image.numpy(), cv2.COLOR_BGR2RGB)
plt.imshow(resize_and_colored.astype(int))
plt.show()

prediction = model.predict(np.array([resized_image]) / 255) 
index = np.argmax(prediction)   # Getting the index of the neuron with the max value

print(f"Predicted class is {class_names[index]}")
