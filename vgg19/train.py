"""
Now in this file model compilation and fittings are set.

Customized parameters :

Opimizer : Adam (leraning_rate = 1e-4)
Loss : Categorical Cross entropy
Metrics : Accuracy
Epochs : 20
Batch Size : 32
Verbose : 1
Callbacks : Model Check Point (saving the best Model)




Codes are created by ©️ Sagnik Roy 2021 .
Do share your feedbacks.

"""


# Libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input , Flatten , Dense
)

# Hyperparametrs
BATCH_SIZE = 32
EPOCHS = 20
VERBOSE = 1


# Model Compilation
clf.compile(
    optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-4 ) ,
    loss = 'categorical_crossentropy' ,
    metrics = ['accuracy']
)

# Callbacks
callbacks = [ keras.callbacks.ModelCheckpoint('animal_classifier.h5' , save_best_only = True) ]


# Model Training
history = clf.fit(
    train_data , 
    epochs = EPOCHS ,
    batch_size = BATCH_SIZE,
    validation_data = validation_data,
    verbose = VERBOSE,
    callbacks = callbacks
)

print('Model has been trained...................')
