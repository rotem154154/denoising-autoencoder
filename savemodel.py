from __future__ import print_function
from Tkinter import *
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.callbacks import TensorBoard
from time import time
import random
import numpy as np

drawpixels2 = []

indexupdate = 0
size = 10
def addnoise(data,prob):
  for k in range(data.shape[0]):
    for i in range(data.shape[1]):
      for j in range(data.shape[2]):
        if random.random() <= prob:
          data[k][i][j] = 1 if data[k][i][j] == 0 else 0
def drawcanvas():
  for i in range(29):
    widget.create_line(size, (i+1) * size, 280 + size, (i+1) * size, width=1)
    widget.create_line((i + 1) * size, size, (i+1) * size, 280 + size, width=1)
    widget.create_line(size+320, (i+1) * size, 280 + size+320, (i+1) * size, width=1)
    widget.create_line((i + 1) * size+320, size, (i+1) * size+320, 280 + size, width=1)
    widget.create_line(size+640, (i+1) * size, 280 + size+640, (i+1) * size, width=1)
    widget.create_line((i + 1) * size+640, size, (i+1) * size+640, 280 + size, width=1)
  text1 = widget.create_text(145, 320, text="Original",font=("Courier", 24))
  text2 = widget.create_text(475, 320, text="Noisy",font=("Courier", 24))
  text3 = widget.create_text(792, 320, text="Denoised",font=("Courier", 24))
def drawimage(img,posx):
  img = img.reshape((28,28))

  img = np.rot90(img)
  for i in range(28):
    for j in range(28):
      if img[i][j] > 0.5:
        drawpixels2.append(
          widget.create_rectangle(((28-i) + 1) * size + posx, (j + 1) * size, ((28-i) + 2) * size + posx, (j + 2) * size,fill='black'))



batch_size = 128
epochs = 32
savemodel = True
modelname = "model1"
noise = 0.24
smalldataset = False

(x_train, _), (x_test, _) = mnist.load_data()
if smalldataset:
  x_train = x_train[:10000]
  x_test = x_test[:1000]
if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, 28, 28)
    x_test = x_test.reshape(x_test.shape[0], 1, 28, 28)
    input_shape = (1, 28, 28)
else:
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    input_shape = (28, 28, 1)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
x_train = np.ceil(x_train)
x_test = np.ceil(x_test)



print(x_train.shape)
print("adding noise")
y_train = np.copy(x_train)
addnoise(y_train,noise)
y_test = np.copy(x_test)
addnoise(y_test,noise)
x_train = x_train.reshape(x_train.shape[0],784)
x_test = x_test.reshape(x_test.shape[0],784)
y_train = y_train.reshape(y_train.shape[0],784)
y_test = y_test.reshape(y_test.shape[0],784)

x_train,y_train = y_train,x_train
x_test,y_test = y_test,x_test


model = Sequential()
model.add(Dense(128,activation='relu',input_shape=(784,)))
model.add(Dense(64,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(128,activation='relu'))
model.add(Dense(784,activation='sigmoid'))
print(x_train.shape)
print(y_train.shape)
model.compile(loss='binary_crossentropy',optimizer='adadelta',metrics=['accuracy'])
tensorboard = TensorBoard(log_dir="logs/{}".format(time()))
model.fit(x_train,y_train,batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(x_test,y_test),callbacks=[tensorboard])
#score = model.evaluate(x_test,y_test,verbose=0)





def reset(x):
  if x == 2:
    global drawpixels2
    for pixel in drawpixels2:
      widget.delete(pixel)
    drawpixels2 = []

def updateimg():
  global indexupdate
  prediction = model.predict(x_test[indexupdate].reshape(1, 784))
  print(prediction)
  reset(2)
  drawimage(x_test[indexupdate], 310)
  drawimage(prediction,630)
  drawimage(y_test[indexupdate], -10)
  tkroot.after(3000,updateimg)
  indexupdate+=1



tkroot = Tk()
canvas_width = 935
canvas_height = 340
widget = Canvas(tkroot,width=canvas_width,height=canvas_height)
widget.pack(expand=YES, fill=BOTH)
drawcanvas()
tkroot.title('Denoiser')
updateimg()
tkroot.mainloop()













