

#Data Preprocessing
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Activation, Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import np_utils
from matplotlib import pyplot as plt
import random

images_dir = os.path.join(os.getcwd(), 'images')
image_size = 100

label_dict = {'with_mask': 1,
              'without_mask': 0
              }
labels = [key for key in label_dict.keys()]

image_data = []
target_data = []



print("DATA PREPROCESSING...")
for label in labels:
    # folder with mask/without_mask images
    os.chdir(images_dir)
    label_folder = os.path.join(images_dir, label)
    image_names = os.listdir(os.path.join(images_dir, label))
    counter = 0
    for image_name in image_names:
        image_loc = os.path.join(label_folder, image_name)
        image = cv2.imread(image_loc)

        try:
            # grayscale
            image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            #resize
            image_resize = cv2.resize(image_gray, (image_size, image_size))
            image_data.append(image_resize)
            target_data.append(label_dict[label])

        except Exception as e:
            print('Exception: ', e)

        if counter % 1000 == 0:
            print("Images Processed: " + str(counter))

        counter += 1

print("DATA PREPROCESSING FINISHED...")

data = np.array(image_data)/255.0
data = np.reshape(data, (data.shape[0], image_size, image_size, 1))
target_data = np.array(target_data)
target_data = np_utils.to_categorical(target_data)

print(data.shape[1:])
print(len(data))


#Neural Network Architecture - Model 1
model = Sequential()

#add first set of layers
model.add(Conv2D(filters=32, kernel_size=(3,3), input_shape=data.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

#add second set of layers
model.add(Conv2D(filters=64, kernel_size=(3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

#flatten, dropout
model.add(Flatten())
model.add(Dropout(rate=0.5))
model.add(Dense(units=32, activation='relu'))
model.add(Dense(units=2, activation='softmax'))





#Neural Network Architecture - Model 2
model_2 = Sequential()

#add first set of layers
model_2.add(Conv2D(filters=32, kernel_size=(3,3), input_shape=data.shape[1:]))
model_2.add(Activation('relu'))
model_2.add(MaxPooling2D(pool_size=(2,2)))

#add second set of layers
model_2.add(Conv2D(filters=64, kernel_size=(3,3)))
model_2.add(Activation('relu'))
model_2.add(MaxPooling2D(pool_size=(2,2)))

#add third set of layers
model_2.add(Conv2D(filters=64, kernel_size=(3,3)))
model_2.add(Activation('relu'))
model_2.add(MaxPooling2D(pool_size=(2,2)))

#flatten, dropout
model_2.add(Flatten())
model_2.add(Dropout(rate=0.5))
model_2.add(Dense(units=32, activation='relu'))
model_2.add(Dense(units=2, activation='softmax'))





#Neural Network Architecture - Model 3
model_3 = Sequential()

#add first set of layers
model_3.add(Conv2D(filters=32, kernel_size=(3,3), input_shape=data.shape[1:]))
model_3.add(Activation('relu'))
model_3.add(MaxPooling2D(pool_size=(2,2)))

#add second set of layers
model_3.add(Conv2D(filters=64, kernel_size=(3,3)))
model_3.add(Activation('relu'))
model_3.add(MaxPooling2D(pool_size=(2,2)))

#add third set of layers
model_3.add(Conv2D(filters=128, kernel_size=(3,3)))
model_3.add(Activation('relu'))
model_3.add(MaxPooling2D(pool_size=(2,2)))

#add fourth set of layers
model_3.add(Conv2D(filters=128, kernel_size=(3,3)))
model_3.add(Activation('relu'))
model_3.add(MaxPooling2D(pool_size=(2,2)))

#flatten, dropout
model_3.add(Flatten())
model_3.add(Dropout(rate=0.5))
model_3.add(Dense(units=32, activation='relu'))
model_3.add(Dense(units=2, activation='softmax'))



#compile Models
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model_2.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model_3.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#set seed
#random.seed(123456)

#Train test split
train_image, test_image, train_target, test_target = train_test_split(data, target_data, test_size=0.20)
print("Training data size: " + str(len(train_image)))
print("Testing data size before Valid split: " + str(len(test_image)))
test_image, valid_image, test_target, valid_target = train_test_split(test_image, test_target, test_size=0.20)

#sizing
print("Testing data size after Valid split: " + str(len(test_image)))
print("Validation data size :" + str(len(valid_image)))


es = EarlyStopping(monitor='val_loss', mode='min', verbose=1)
mc = ModelCheckpoint('model-{epoch:03d}.model', monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
history = model.fit(train_image, train_target, epochs=15, validation_data=(test_image, test_target), callbacks=[es, mc])

es_2 = EarlyStopping(monitor='val_loss', mode='min', verbose=1)
mc_2 = ModelCheckpoint('model_2-{epoch:03d}.model', monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
history_2 = model_2.fit(train_image, train_target, epochs=15, validation_data=(test_image, test_target), callbacks=[es_2, mc_2])

es_3 = EarlyStopping(monitor='val_loss', mode='min', verbose=1)
mc_3 = ModelCheckpoint('model_3-{epoch:03d}.model', monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
history_3 = model_3.fit(train_image, train_target, epochs=15, validation_data=(test_image, test_target), callbacks=[es_2, mc_2])


#Validation Data
print(model.evaluate(valid_image, valid_target))
print(model_2.evaluate(valid_image, valid_target))
print(model_3.evaluate(valid_image, valid_target))

plt.figure(1)
plt.plot(history.history['loss'], 'g', label='Training Loss - Model 1')
plt.plot(history.history['val_loss'], 'r', label='Test Loss - Model 1')
plt.plot(history_2.history['loss'], 'b', label='Training Loss - Model 2')
plt.plot(history_2.history['val_loss'], 'y', label='Test loss - Model 2')
plt.plot(history_3.history['loss'], 'c', label='Training Loss - Model 3')
plt.plot(history_3.history['val_loss'], 'k', label='Test loss - Model 3')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


plt.figure(2)
plt.plot(history.history['accuracy'], 'g', label='Training Accuracy - Model 1')
plt.plot(history.history['val_accuracy'], 'r', label='Test Accuracy - Model 1')
plt.plot(history_2.history['accuracy'], 'b', label='Training Accuracy - Model 2')
plt.plot(history_2.history['val_accuracy'], 'y', label='Test Accuracy - Model 2')
plt.plot(history_3.history['accuracy'], 'c', label='Training Accuracy - Model 3')
plt.plot(history_3.history['val_accuracy'], 'k', label='Test Accuracy - Model 3')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()





trained_face_data = cv2.CascadeClassifier(cv2.data.haarcascades +  'haarcascade_frontalface_default.xml')

webcam = cv2.VideoCapture(0)

label_dict = {1: 'with_mask',
              0: 'without_mask'
              }

while True:
    # read current frame
    successful_frame_read, frame = webcam.read()

    # convert frame to grayscale
    grayscale_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect faces
    face_coordinates = trained_face_data.detectMultiScale(grayscale_img)

    # draw rectangle around faces
    for (x, y, w, h) in face_coordinates:
        face_img = grayscale_img[y:y+h, x:x+w]
        face_img_resize = cv2.resize(face_img, (100, 100))
        face_img_normalized = face_img_resize/255.0
        face_img_reshape = np.reshape(face_img_normalized, (1,100,100,1))
        face_img_result = model_3.predict(face_img_reshape)

        label = np.argmax(face_img_result, axis=1)[0]

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, label_dict[label], (x, y-10), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.8, (255, 255, 255), 2)

    # return frame
    cv2.imshow('Face detector', frame)
    key = cv2.waitKey(1)

    # quit loop
    if key == 81 or key == 113:
        webcam.release()
        break
