# Sword Detection and Segmentation

This is an implementation of [Mask R-CNN](https://arxiv.org/abs/1703.06870) on Python 3, Keras, and TensorFlow. The model generates bounding boxes and segmentation masks for each instance of a gun in the image. It's based on Feature Pyramid Network (FPN) and a ResNet101 backbone.

I have trained the above model for detection and segmentation of swords in a given image. 

Steps followed:
1. Collected 300 swords containing image data from google.
2. Created json file by locating the swords in an image as a groundtruth.
3. Divide the dataset into train and test in the ratio of 80:20. 
4. Train the model using train dataset.
5. Then test the model which gives nice result by locating the swords in the given image.
