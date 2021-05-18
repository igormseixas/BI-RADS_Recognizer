import tensorflow as tf
import numpy as np

from tensorflow import keras
#from tensorflow.keras.layers import Dense, Flatten, Conv2D
#from tensorflow.keras import Model, Sequential

print(tf.__version__)


# Function to read the Sample file.
def readSample(file_name):
    # Variables.
    index = []
    data = []
    BIRADS = []
    
    # Open file.
    sample_file = open(file_name, "r")
    
    # Read all file.
    for line in sample_file:
        # Read the first line.
        # line = sample_file.readline()
        line = line.split(sep=",", maxsplit=1) # Split is necessary to get negative indexes.
        # Get index in a string.
        index.append(line[0]) 
        # Get that in a string and transform to a NumPy Array of float64. Line at postion[1] will get all values of the image characterisc, plus its BIRADS.
        raw_data = (line[1][1:-4])
        nprawdata = np.fromstring(raw_data, dtype=np.float64, sep=',')
        #print(nprawdata)
        #print(len(nprawdata))
        data.append(nprawdata)
        # nptmp = np.array(npdata, index)
        # Get BIRADS information. Same thing of the raw_data to position[1]
        BIRADS.append(int(line[1][-2]))

    # Convert to np array.
    npindex = np.array(index)
    npdata = np.array(data)
    npBIRADS = np.array(BIRADS, dtype=np.uint8)

    # Close file.
    sample_file.close()

    # Return the triple.
    return (npindex, npdata, npBIRADS)

# Create the train and test data set.
(index, train_images, train_labels) = readSample("training_file6")
(index, test_images, test_labels) = readSample("scratch_test_file2")

#(index, train_images, train_labels) = readSample("training_file")
#(index, test_images, test_labels) = readSample("test_file")

print("Train Images shape: ", train_images.shape)
print("Train Images dtype: ", train_images.dtype)
print("Characterist of image 1: ", train_images[0])
print("Train Labels Length: ", len(train_labels))
print(train_labels)

# Create and build the model.
model = keras.Sequential([
    keras.layers.Dense(128, activation='relu', input_shape=(120,)),
    keras.layers.Dropout(.1),
    keras.layers.Dense(128, activation='relu'),
    #keras.layers.Dropout(.1),
    keras.layers.Dense(128, activation='relu'),
    #keras.layers.Dropout(.1),
    keras.layers.Dense(128, activation='relu'),
    #keras.layers.Dropout(.1),
    keras.layers.Dense(128, activation='relu'),
    #keras.layers.Dropout(.1),
    keras.layers.Dense(4, activation='softmax')
])

model.summary()

# Compile model.
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# Train model.
model.fit(train_images, train_labels, epochs=200)

# Check the accuracy.
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)

# Predition. # Test with 10, 15.
prediction = model.predict(test_images)
print(prediction[2])
print(np.argmax(prediction[2]))
print(test_labels[2])