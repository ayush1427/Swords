# Sword Detection and Segmentation
I have trained a MASK-RCNN model for detection and segmentation of swords in a given image. 
Steps followed:
1. Collected 300 swords containing image data from google.
2. Created json file by locating the swords in an image as a groundtruth.
3. Divide the dataset into train and test in the ratio of 80:20. 
4. Train the model using train dataset.
5. Then test the model which gives nice result by locating the swords in the given image.
