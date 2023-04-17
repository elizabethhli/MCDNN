import numpy as np
import torch
import torch.nn as tnn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn.functional as F
from torch.autograd import Variable

# import tensorflow as tf

from keras.preprocessing.image import ImageDataGenerator

from mcdnn import mcdnn

EPOCH = 800
BATCH_SIZE = 100
LEARNING_RATE = 0.001

# # Load the MNIST dataset
# transform = transforms.Compose([transforms.Resize((28, 28)), transforms.ToTensor()])
# train_data = dsets.MNIST(root='MNIST_data/', train=True, transform=transform, download=True)
# test_data = dsets.MNIST(root='MNIST_data/', train=False, transform=transform, download=True)

# train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
# test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=False)

# Load the PASCAL VOC dataset
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
train_data = dsets.VOCDetection(root='VOCdevkit/', year='2007', image_set='train', transform=transform)
test_data = dsets.VOCDetection(root='VOCdevkit/', year='2007', image_set='test', transform=transform)

train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=False)

mymodel = mcdnn()
mymodel.cuda()

cost = tnn.CrossEntropyLoss()

for epoch in range(EPOCH):
    avg_cost = 0
    add_iter = 0
    total_batch = len(train_loader)

    LEARNING_RATE = max(0.00003, (LEARNING_RATE * 0.993))
    print('current learning rate: %f' % LEARNING_RATE)
    for i, (batch_x, batch_y) in enumerate(train_loader):
        # batch_x = batch_x.view(BATCH_SIZE, 1, 28, 28)
        batch_x = batch_x.view(BATCH_SIZE, 3, 224, 224) # Update the input size and channels
        batch_y = batch_y

        images = Variable(batch_x).cuda()
        labels = Variable(batch_y).cuda()

        optimizer = torch.optim.Adam(mymodel.parameters(), lr=LEARNING_RATE)
        optimizer.zero_grad()
        model_output = mymodel.forward(images)
        loss = cost(model_output, labels)
        avg_cost += loss.item()
        loss.backward()
        optimizer.step()

        # Data augmentation in every 4 steps
        if ((i+1) % 4 == 0):
            # ZCA whitening
            if ((i+1) % 16 == 0) :
              print('ZCA whitening') 
              datagen = ImageDataGenerator(zca_whitening=True)
            # Feature Standardization
            elif ((i+1) % 12 == 0) :
              print('Feature Standardization') 
              datagen = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True)
            # Random Rotation up to 90 degrees
            elif ((i+1) % 8 == 0) :
              print('Random Rotation') 
              datagen = ImageDataGenerator(rotation_range=90)
            # Random shift
            else :
              print('Random Shift') 
              datagen = ImageDataGenerator(width_shift_range=0.2, height_shift_range=0.2)
            
            # batch_x = np.reshape(batch_x, (BATCH_SIZE,28,28,1))
            batch_x = np.reshape(batch_x, (BATCH_SIZE, 224, 224, 3))  # Update the input size and channels
            datagen.fit(batch_x)
            for aug_batch_x in datagen.flow(batch_x, batch_size=BATCH_SIZE):
              # aug_batch_x = np.reshape(aug_batch_x, (BATCH_SIZE,1,28,28))
              aug_batch_x = np.reshape(aug_batch_x, (BATCH_SIZE, 3, 224, 224))  # Update the input size and channels
              images = Variable(torch.Tensor(aug_batch_x)).cuda()

              optimizer = torch.optim.Adam(mymodel.parameters(), lr=LEARNING_RATE)
              optimizer.zero_grad()
              model_output = mymodel.forward(images)
              loss = cost(model_output, labels)
              avg_cost += loss.item()
              # avg_cost += loss.data[0]
              add_iter += 1
              loss.backward()
              optimizer.step()
              break
    if ((i+1) % 1 == 0):
        print('Epoch [%d/%d], Iter[%d/%d] avg Loss. %.4f' %
              (epoch+1, EPOCH, i+1, total_batch, avg_cost/(add_iter + i + 1)))

    test_batch = len(test_loader)
    accuracy = 0

    for i, (test_x, test_y) in enumerate(test_loader):
        test_x = test_x.view(test_x.size(0), 3, 224, 224)  # Update the input size and channels
        # test_x = test_x.view(test_x.size(0), 1, 28, 28)
        test_y = test_y

        test_images = Variable(test_x).cuda()
        test_labels = test_y

        test_output = mymodel(test_images)
        test_output = np.argmax(test_output.cpu().data.numpy(), axis=1)

        accuracy_temp = float(np.sum(test_labels.numpy() == test_output))
        accuracy += accuracy_temp
        if ((i+1) % 10 == 0):
            print("Epoch [%d/%d], TestBatch [%d/%d] batch acc: %f" %
                  (epoch+1, EPOCH, i+1, test_batch, (accuracy_temp/100)))

    print("Accuracy: %f" % (accuracy/len(test_data)))