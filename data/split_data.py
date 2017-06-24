#!/usr/bin/python

import os, sys, glob
from shutil import copyfile, rmtree
import numpy as np

# Create train directory and subdirectories
os.mkdir("train/")
os.mkdir("train/vehicles/")
os.mkdir("train/non-vehicles/")
# Create validation directory and subdirectories
os.mkdir("validation/")
os.mkdir("validation/vehicles/")
os.mkdir("validation/non-vehicles/")
# Create test directory and subdirectories
os.mkdir("test/")
os.mkdir("test/vehicles/")
os.mkdir("test/non-vehicles/")
# Create sample directory and subdirectories
os.mkdir("sample/")
os.mkdir("sample/train/")
os.mkdir("sample/validation/")
os.mkdir("sample/test/")
os.mkdir("sample/train/vehicles/")
os.mkdir("sample/train/non-vehicles/")
os.mkdir("sample/validation/vehicles/")
os.mkdir("sample/validation/non-vehicles/")
os.mkdir("sample/test/vehicles/")
os.mkdir("sample/test/non-vehicles/")

# Randomly split vehicle data into train/validation/test sets
vehicle_dir = 'vehicles/'
img_dirs = os.listdir(vehicle_dir)
vehicles = []
for img_dir in img_dirs:
    vehicles.extend(glob.glob(vehicle_dir+img_dir+'/*'))
np.random.seed(42)
shuf = np.random.permutation(vehicles)
train = shuf[:int(len(shuf)*0.8)]
val = shuf[int(len(shuf)*0.8):int(len(shuf)*0.9)]
test = shuf[int(len(shuf)*0.9):]
for i in range(len(train)):
    os.rename(train[i], 'train/'+vehicle_dir+str(i)+'.png')

for i in range(len(val)):
    os.rename(val[i], 'validation/'+vehicle_dir+str(i+len(train))+'.png')

for i in range(len(test)):
    os.rename(test[i], 'test/'+vehicle_dir+str(i+len(train)+len(val))+'.png')

# Randomly split non-vehicle data into train/validation/test sets
non_vehicle_dir = 'non-vehicles/'
img_dirs = os.listdir(non_vehicle_dir)
non_vehicles = []
for img_dir in img_dirs:
    non_vehicles.extend(glob.glob(non_vehicle_dir+img_dir+'/*'))
np.random.seed(42)
shuf = np.random.permutation(non_vehicles)
train = shuf[:int(len(shuf)*0.8)]
val = shuf[int(len(shuf)*0.8):int(len(shuf)*0.9)]
test = shuf[int(len(shuf)*0.9):]
for i in range(len(train)):
    os.rename(train[i], 'train/'+non_vehicle_dir+str(i)+'.png')

for i in range(len(val)):
    os.rename(val[i], 'validation/'+non_vehicle_dir+str(i+len(train))+'.png')

for i in range(len(test)):
    os.rename(test[i], 'test/'+non_vehicle_dir+str(i+len(train)+len(val))+'.png')

# Copy random data to create a small sample dataset
g = glob.glob('train/vehicles/*')
np.random.seed(42)
shuf = np.random.permutation(g)
for i in range(200):
    parts = shuf[i].split('/')
    copyfile(shuf[i], 'sample/train/vehicles/'+parts[2])

g = glob.glob('train/non-vehicles/*')
np.random.seed(42)
shuf = np.random.permutation(g)
for i in range(200):
    parts = shuf[i].split('/')
    copyfile(shuf[i], 'sample/train/non-vehicles/'+parts[2])

g = glob.glob('validation/vehicles/*')
np.random.seed(42)
shuf = np.random.permutation(g)
for i in range(50):
    parts = shuf[i].split('/')
    copyfile(shuf[i], 'sample/validation/vehicles/'+parts[2])

g = glob.glob('validation/non-vehicles/*')
np.random.seed(42)
shuf = np.random.permutation(g)
for i in range(50):
    parts = shuf[i].split('/')
    copyfile(shuf[i], 'sample/validation/non-vehicles/'+parts[2])

g = glob.glob('test/vehicles/*')
np.random.seed(42)
shuf = np.random.permutation(g)
for i in range(50):
    parts = shuf[i].split('/')
    copyfile(shuf[i], 'sample/test/vehicles/'+parts[2])

g = glob.glob('test/non-vehicles/*')
np.random.seed(42)
shuf = np.random.permutation(g)
for i in range(50):
    parts = shuf[i].split('/')
    copyfile(shuf[i], 'sample/test/non-vehicles/'+parts[2])

# Remove the empty vehicle and non-vehicle top directories
rmtree('vehicles/')
rmtree('non-vehicles/')
