import numpy as np 
import keras
from sklearn.model_selection import train_test_split
from architectures import *
import argparse
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='Axionaut training')
parser.add_argument('--augmentation', default=True)
parser.add_argument('--val_split', default=0.2)
parser.add_argument('--epochs', default=100, type=int)
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--early_stop', default=True)
parser.add_argument('--patience', default=5, type=int)

args = parser.parse_args()

# Model path
model_path = 'Models/' # Put your own path.

# Get training data from the Axionable track
X_axio = np.load('Datasets/axionable_data/X_train_axio.npy') # Put the path of your data.
Y_axio = np.load('Datasets/axionable_data/Y_train_axio.npy')
print('Shape = ', np.shape(X_axio))

# Create autopilot model from architectures and print summary
model =  model_categorical(input_size=(130,250,3), dropout=0.1)
model.summary()

# Train model
model_name = model_path + 'checkpoint_model.hdf5'
min_delta=.0005

#checkpoint to save model after each epoch
save_best = keras.callbacks.ModelCheckpoint(model_name, 
                                            monitor='val_loss', 
                                            verbose=1, 
                                            save_best_only=True, 
                                            mode='min')

#stop training if the validation error stops improving.
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', 
                                           min_delta=min_delta, 
                                           patience=args.patience, 
                                           verbose=1, 
                                           mode='auto')

callbacks_list = [save_best]

if args.early_stop:
    callbacks_list.append(early_stop)

hist = model.fit(
                X_train, 
                Y_train,
                nb_epoch=args.epochs,
                batch_size=args.batch_size, 
                verbose=1, 
                validation_data=(X_val, Y_val),
                callbacks=callbacks_list,
                shuffle=True)


model.save('final_model.hdf5') # Put the name of your choice.