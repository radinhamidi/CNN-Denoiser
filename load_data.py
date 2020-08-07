'''

Radin Hamidi Rad
500979422

'''

from util import *
import cv2 as cv

data_path = './dataset/'
train_x_name = 'train_X.bin'
train_y_name = 'train_y.bin'
test_x_name = 'test_x.bin'
test_y_name = 'test_y.bin'
unlableled_x_name = 'unlabeled_X.bin'
size = 3 * 96 * 96
size_50 = 3 * 50 * 50
size_72 = 3 * 72 * 72

# Train_X
train_x = np.fromfile(data_path+train_x_name, dtype=np.uint8)
images_train = np.reshape(train_x, (-1, 3, 96, 96))
images_train = np.transpose(images_train, (0, 3, 2, 1))

images_train_50 = []
for i in images_train:
    images_train_50.append(cv.resize(i, (50,50)))
images_train_50 = np.reshape(images_train_50, (-1, size_50))

images_train_72 = []
for i in images_train:
    images_train_72.append(cv.resize(i, (72,72)))
images_train_72 = np.reshape(images_train_72, (-1, size_72))

images_train = np.reshape(images_train, (-1, size))
np.save(data_path+'train_x_50', images_train_50)
np.save(data_path+'train_x_72', images_train_72)
np.save(data_path+'train_x', images_train)
print('Train_X Done.')


# Train_Y
train_y = np.fromfile(data_path+train_y_name, dtype=np.uint8)
np.save(data_path+'train_y', train_y)
print('Train_Y Done.')


# Test_X
test_x = np.fromfile(data_path+test_x_name, dtype=np.uint8)
images_test = np.reshape(test_x, (-1, 3, 96, 96))
images_test = np.transpose(images_test, (0, 3, 2, 1))

images_test_50 = []
for i in images_test:
    images_test_50.append(cv.resize(i, (50,50)))
images_test_50 = np.reshape(images_test_50, (-1, size_50))

images_test_72 = []
for i in images_test:
    images_test_72.append(cv.resize(i, (72,72)))
images_test_72 = np.reshape(images_test_72, (-1, size_72))

images_test = np.reshape(images_test, (-1, size))
np.save(data_path+'test_x_50', images_test_50)
np.save(data_path+'test_x_72', images_test_72)
np.save(data_path+'test_x', images_test)
print('Test_X Done.')


# Test_Y
test_y = np.fromfile(data_path+test_y_name, dtype=np.uint8)
np.save(data_path+'test_y', test_y)
print('Test_Y Done.')


# Unlabeled_X
unlableled_x = np.fromfile(data_path+unlableled_x_name, dtype=np.uint8)
unlableled_x = np.reshape(unlableled_x, (-1, 3, 96, 96))
unlableled_x = np.transpose(unlableled_x, (0, 3, 2, 1))

unlableled_x_50 = []
for i in unlableled_x:
    unlableled_x_50.append(cv.resize(i, (50,50)))
unlableled_x_50 = np.reshape(unlableled_x_50, (-1, size_50))

unlableled_x_72 = []
for i in unlableled_x:
    unlableled_x_72.append(cv.resize(i, (72,72)))
unlableled_x_72 = np.reshape(unlableled_x_72, (-1, size_72))

unlableled_x = np.reshape(unlableled_x, (-1, size))
np.save(data_path+'unlabeled_x_50', unlableled_x_50)
np.save(data_path+'unlabeled_x_72', unlableled_x_72)
np.save(data_path+'unlabeled_x', unlableled_x)
print('Unlableled_x Done.')