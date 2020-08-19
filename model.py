import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Activation, MaxPooling2D, Flatten, BatchNormalization, Dropout
from keras.optimizers import Adam
import sklearn.model_selection as model_selection
from keras.preprocessing.image import ImageDataGenerator

# hyperparameters
lr = 0.001
filters = 3
pool_size= (3,3)
kernel_size= (3,3)
n_layers = 3
epochs = 20
batch_size = 32

def append_ext(filename):
    return filename+".jpg"

# train
train_df = pd.read_csv("./data/Datasets/csv/train.csv")
train_df["image_name"]=train_df["image_name"].apply(append_ext)

# test
# test_df = pd.read_csv("./data/Datasets/csv/test.csv")

# image preprocessing
# train_dir = "~/PycharmProjects/melanoma_project/data/Datasets/jpeg/train"
# test_dir =  "~/PycharmProjects/melanoma_project/data/Datasets/jpeg/test"

train_dir = "./data/Datasets/jpeg/train"
test_dir =  "./data/Datasets/jpeg/test"
X_train, X_test = model_selection.train_test_split(
    train_df,
    train_size=0.2,
    test_size=0.1,
    random_state=8
)

X_train, X_validation = model_selection.train_test_split(
    X_train,
    train_size=0.75,
    test_size=0.25,
    random_state=8
)

image_gen = ImageDataGenerator(
    brightness_range=None,
    channel_shift_range=0.0,
    fill_mode="nearest",
    cval=0.0,
    rescale=1./255,
    preprocessing_function=None,
    dtype=None
)

X_train.target = X_train.target.map({0:'normal', 1: 'melanoma'})
train_generator = image_gen.flow_from_dataframe(
    dataframe=X_train,
    directory=train_dir,
    x_col="image_name",
    y_col="target",
    target_size=(256, 256),
    color_mode="rgb",
    class_mode="categorical",
    batch_size=batch_size,
    seed=25,
)

X_validation.target = X_validation.target.map({0:'normal', 1: 'melanoma'})
valid_generator = image_gen.flow_from_dataframe(
    dataframe=X_validation,
    directory=train_dir,
    x_col="image_name",
    y_col="target",
    target_size=(256, 256),
    color_mode="rgb",
    class_mode="categorical",
    batch_size=batch_size,
    seed=25,
)

X_test.target = X_test.target.map({0:'normal', 1: 'melanoma'})
test_generator = image_gen.flow_from_dataframe(
    dataframe=X_test,
    directory=train_dir,
    x_col="image_name",
    y_col="target",
    target_size=(256, 256),
    color_mode="rgb",
    class_mode="categorical",
    batch_size=batch_size,
    seed=25,
)

# neural network
model = Sequential()
model.add(Conv2D(filters=filters,kernel_size=(3,3),padding='same',input_shape=[256,256,3])) # add input shape argument, Convolution2D?
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=pool_size))
 for i in range(n_layers-1):
   model.add(Conv2D(filters=filters,kernel_size=(3,3),padding='same'))
   model.add(Activation('relu'))
   model.add(MaxPooling2D(pool_size=pool_size))
model.add(Flatten())
model.add(Dense(2))
model.add(Activation('softmax')) # maybe try using tanh, it's a binary classification problem
print(model.summary())

optimizer = Adam(learning_rate=lr)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

model.fit_generator(train_generator,
                    epochs = epochs,
                    validation_data = valid_generator)
#model.save_weights('weights.h5')
