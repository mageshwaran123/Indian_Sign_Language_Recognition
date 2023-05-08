# Indian_Sign_Language_Recognition

DataSet - Indian Sign Language (IST) - https://www.kaggle.com/datasets/prathumarikeri/indian-sign-language-isl
Note: Split the DataSet into Train (80%) and Test (20%)

#Pre_Notebook is a summary of a Python program that trains a deep learning model using the ResNet50 architecture on the Indian hand sign dataset. Here is a breakdown of the steps involved:

Step 1 - Load the Required packages: This step imports the necessary packages, including TensorFlow and Keras, for building and training the model.

Step 2 - Load the Indian hand Sign dataset: The code specifies the directories for the training and testing data and uses the ImageDataGenerator class from TensorFlow to preprocess the images. It also creates data generators for both the training and testing datasets.

Step 3 - Load the pre-Trained ResNet50 Model: The code loads the pre-trained ResNet50 model from Keras, with the weights pre-trained on the ImageNet dataset.

Step 4 - Freeze the weight of the Pre_trained layers: This step freezes the weights of the pre-trained layers in the ResNet50 model to prevent them from being updated during training.

Step 5 - Add a new classification layer on top of the Pre-Trained Model: The code adds additional layers on top of the pre-trained ResNet50 model to adapt it for the specific hand sign classification task. This includes a global average pooling layer and two dense layers with ReLU and softmax activations, respectively.

Step 6 - Define New Model: The code defines a new model by specifying the inputs and outputs of the network.

Step 7 - Compile the model with a suitable optimizer and loss function: This step compiles the model, specifying the optimizer (Adam) and loss function (categorical cross-entropy) to be used during training. It also includes accuracy as a metric to monitor.

Step 8 - Train the model on the train data: The code trains the model using the fit function, specifying the number of epochs and the training and validation data generators.

Training accuracy: accuracy value after each epoch. The last epoch shows an accuracy of 0.9753.

Step 9 - Evaluate the model on the test data: After training, the code evaluates the model's performance on the test data, calculating the test loss and accuracy.

Test accuracy: The code evaluates the model on the test data using the evaluate function. The output shows a test accuracy of 0.9861.

Step 10 - Save the Model: Finally, saves the trained model as a file named "model.h5".

summary provides an overview of the code's functionality, which involves loading the dataset, building a pre-trained model, training the model, evaluating its performance, and saving it for future use.

#Post_Notebook - image classification using a pre-trained model. Here is a summary of the code:

- Import necessary packages: The code imports the required packages, including cv2 for image processing, numpy for numerical operations, tensorflow.keras.models for loading the model, and matplotlib.pyplot for displaying images.

- Load the pre-trained model: The code loads a pre-trained model from the file 'model.h5' using the load_model function.

- Define class labels: A dictionary class_labels is defined, mapping numerical class indices to their corresponding labels.

- Define image paths: A list image_paths is created, containing the paths of the images to be classified.

- Iterate over the image paths: The code iterates over each image path in image_paths.

- Load and preprocess the input image: For each image, it is loaded using cv2.imread and converted to RGB color format. The image is then resized to (224, 224) dimensions, normalized to the range [0, 1], and expanded to have a batch dimension.

- Perform prediction: The preprocessed image is fed into the pre-trained model using the predict function, obtaining a prediction output. The predicted class index is determined as the index of the maximum value in the prediction output.

- Display the predicted image and label: The image is displayed using matplotlib.pyplot.imshow, along with the predicted label as the title of the plot.

- Print the predicted class label: The image path and predicted label are printed to the console.

The code processes multiple images, loads the model, performs predictions, displays the images, and prints the predicted labels for each image.
