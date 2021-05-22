'''
/**
* 1/2021
* Pontifícia Universidade Católica de Minas Gerais
* Processamento de Imagens - Prof. Alexei Machado
* Designed by:
* @author Gabriel Antonio Diniz Meneguite
* @author Igor Machado Seixas
* @author Matheus Mendes Tavares
* @version 1.00a
*/
'''

import tensorflow as tf
import numpy as np
import sys

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
'''
0 = all messages are logged (default behavior)
1 = INFO messages are not printed
2 = INFO and WARNING messages are not printed
3 = INFO, WARNING, and ERROR messages are not printed
'''

from tensorflow import keras
#from tensorflow.keras.layers import Dense, Flatten, Conv2D
#from tensorflow.keras import Model, Sequential

print()
print(tf.__version__)

# Define a Custom Callback to set a max loss or a accuracy.
class CustomCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        #if logs.get('accuracy') >= 9e-1:
        #if logs.get('accuracy') >= 1:
        if logs.get('loss') <= 1e-2: #0.01
            self.model.stop_training = True

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
        split_raw_data = raw_data.split(sep=",")
        # Get data and separate through characteristics.
        homogeneity = split_raw_data[0:120]
        entropy = split_raw_data[120:240]
        contrast = split_raw_data[240:360]
 
        # Select what characteristics do tou what to feed the neural network.
        #final_data = np.array(homogeneity+entropy+contrast, dtype=np.float64)
        '''
        homogeneity - entropy
        homogeneity - contrast
        homogeneity - entropy, contrast
        entropy - contrast
        '''
        final_data = np.array(homogeneity+entropy, dtype=np.float64)
        
        #nprawdata = np.fromstring(raw_data, dtype=np.float64, sep=',')
        nprawdata = final_data
        data.append(nprawdata)

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

# Function to read and classify the region file.
def classifyRegionSample(model):
    # Open file.
    region_sample_file = open("region_file", "r")

    # The file will contain only the description.
    # Get data in a string and split it
    line = region_sample_file.readline()
    split_raw_data = line[1:-1].split(sep=',')

    '''
    # Get data and separate through characteristics.
    homogeneity = split_raw_data[0:120]
    entropy = split_raw_data[120:240]
    contrast = split_raw_data[240:360]
    '''

    # Select what characteristics do to what to feed the neural network.
    #final_region_data = np.array(homogeneity+entropy+contrast, dtype=np.float64)
    final_region_data = np.array(split_raw_data, dtype=np.float64)
    data = []
    data.append(final_region_data)

    # Add the region data to a batch where it's the only member.
    #region_data = (np.expand_dims(final_region_data,0))
    region_data = np.array(data)

    # Close file.
    region_sample_file.close()
    #print(region_data.shape)

    # Classify the region.
    prediction = model.predict(region_data)
    print("Análise:", prediction[0])
    print("BIRADS que a rede classificou:", np.argmax(prediction[0])+1)

def printConfusionMatrix(model):
    # Variables.
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=0)
    prediction = model.predict(test_images)
    confusion_matrix = np.zeros((4,4))
    count = 0

    print("Confusion Matrix")
    
    # Build the confusion matrix.
    for pred in prediction[0:25]:
        confusion_matrix[0][np.argmax(pred)] += 1
    for pred in prediction[25:50]:
        confusion_matrix[1][np.argmax(pred)] += 1
    for pred in prediction[50:75]:
        confusion_matrix[2][np.argmax(pred)] += 1
    for pred in prediction[75:100]:
        confusion_matrix[3][np.argmax(pred)] += 1
        
    print(confusion_matrix)
    print()

    accuracy = confusion_matrix.diagonal().sum()/100
    print("Accuracy = ", accuracy)
    specificity = 1-((confusion_matrix.sum()-confusion_matrix.diagonal().sum())/300)
    print("Specificity = ",specificity)

# Function to print the BIRADS Accuracy.
def printBIRADSAccuracy():
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=0)
    print("Loss and accuracy with all test set.")
    print("Loss: ", test_loss, " - Accuracy", test_acc)

    biradsI_loss, biradsI_acc       =   model.evaluate(test_images[0:25], test_labels[0:25], verbose=0)
    biradsII_loss, biradsII_acc     =   model.evaluate(test_images[25:50], test_labels[25:50], verbose=0)
    biradsIII_loss, biradsIII_acc   =   model.evaluate(test_images[50:75], test_labels[50:75], verbose=0)
    biradsIV_loss, biradsIV_acc     =   model.evaluate(test_images[75:100], test_labels[75:100], verbose=0)
    
    print("BIRADS I   - Accuracy = ", biradsI_acc)
    print("BIRADS II  - Accuracy = ", biradsII_acc)
    print("BIRADS III - Accuracy = ", biradsIII_acc)
    print("BIRADS IV  - Accuracy = ", biradsIV_acc)

# Function to create and compile model.
def createModel():
    # Create and build the model.
    model = keras.Sequential([
        keras.layers.Dense(128, activation='relu', input_shape=(train_images.shape[1],)), # Data need to be in the same shape of the training data. Ex 240 characteristics 240 shape.
        #keras.layers.Dropout(.1),
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

    # Compile model.
    model.compile(optimizer='adam',
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])

    return model

def trainModel():
    #print("Train Images shape: ", train_images.shape)
    #print("Train Images dtype: ", train_images.dtype)
    #print("Characterist of image 1: ", train_images[0])

    #print("Train Labels Length: ", len(train_labels))
    #print(train_labels)
    
    global best_neural_network
    # Create model.
    model = createModel()

    #'''
    # Train model.
    #callback = tf.keras.callbacks.EarlyStopping(monitor='loss', mode='min', min_delta=0.00000000000001, patience=20)
    callback = CustomCallback()
    model.fit(train_images, train_labels, epochs=10000, verbose=0, callbacks=[callback], use_multiprocessing=True)
    #model.fit(train_images, train_labels, epochs=3000, verbose=2, use_multiprocessing=True)

    # Check the accuracy.
    print('\n'*2)
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=0)
    print("Loss and accuracy with all test set.")
    print("Loss: ", test_loss, " - Accuracy", test_acc)

    biradsI_loss, biradsI_acc       =   model.evaluate(test_images[0:25], test_labels[0:25], verbose=0)
    biradsII_loss, biradsII_acc     =   model.evaluate(test_images[25:50], test_labels[25:50], verbose=0)
    biradsIII_loss, biradsIII_acc   =   model.evaluate(test_images[50:75], test_labels[50:75], verbose=0)
    biradsIV_loss, biradsIV_acc     =   model.evaluate(test_images[75:100], test_labels[75:100], verbose=0)
    
    print("BIRADS I   - Accuracy = ", biradsI_acc)
    print("BIRADS II  - Accuracy = ", biradsII_acc)
    print("BIRADS III - Accuracy = ", biradsIII_acc)
    print("BIRADS IV  - Accuracy = ", biradsIV_acc)

    if(test_acc > best_neural_network):
        # Save the model.
        print("Saving the model...")
        model.save('neural_network_model')
        best_neural_network = test_acc
    
    return test_acc

# Global variables.
# Create the train and test data set.
(index, train_images, train_labels) = readSample("sample_files/shuf_all_descriptions_training")
(index, test_images, test_labels) = readSample("sample_files/all_descriptions_test")
best_neural_network = 0.69

# Check the command to print model.
if(len(sys.argv) > 1 and sys.argv[1] == "print_model"):
    # Load the last model.
    model = tf.keras.models.load_model('neural_network_model')
    # Print the model.
    model.summary()

if(len(sys.argv) > 1 and sys.argv[1] == "train_model"):
    while(1):
        print()
        print("Record: ", best_neural_network)
        if(trainModel()>best_neural_network):
            print("MELHORAMOS A REDE!")
            
    
# Check the command to print confusion matrix.
if(len(sys.argv) > 1 and sys.argv[1] == "print_confusion_matrix"):
    # Load the last model.
    model = tf.keras.models.load_model('neural_network_model')
    # Print the matrix.
    printConfusionMatrix(model)

# Check the command to classify the region.
if(len(sys.argv) > 1 and sys.argv[1] == "classify_region"):
    # Load the last model.
    model = tf.keras.models.load_model('neural_network_model')
    # Classify the region.
    classifyRegionSample(model)

# Check the command to classify the region.
if(len(sys.argv) > 1 and sys.argv[1] == "birads_accuracy"):
    # Load the last model.
    model = tf.keras.models.load_model('neural_network_model')
    # Print BIRADS accuracy.
    printBIRADSAccuracy()
    print()