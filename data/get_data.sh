#!/bin/bash

# Get vehicle data and unzip if not already downloaded
if [ ! -e vehicles ]; then
  wget "https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip"
  unzip vehicles.zip
  rm vehicles.zip
fi

# Get non-vehicle data and unzip if not already downloaded
if [ ! -e non-vehicles ]; then
  wget "https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip"
  unzip non-vehicles.zip
  rm non-vehicles.zip
fi

# Remove unwanted __MACOSX file
if [ -e __MACOSX ]; then
  rm -r __MACOSX
fi
