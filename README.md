# Melanoma Detection Assignment
> Build a multiclass classification model using a custom convolutional neural network in TensorFlow.

> build a CNN based model which can accurately detect melanoma. Melanoma is a type of cancer that can be deadly if not detected early. It accounts for 75% of skin cancer deaths. A solution that can evaluate images and alert dermatologists about the presence of melanoma has the potential to reduce a lot of manual effort needed in diagnosis.


## Table of Contents
* [General Info](#general-information)
* [Technologies Used](#technologies-used)
* [Conclusions](#conclusions)
* [Acknowledgements](#acknowledgements)

<!-- You can include any other section that is pertinent to your problem -->

## General Information
- There are 9 classes and 2239 image for all the classes.
- We need to build a CNN based model from scratch (No Transfer learning) to detect melanoma in a image
- Melonoma detection is critical for healthcare professional as it affects the resolution and treatment time for doctors and patients.
- **Dataset:** https://drive.google.com/file/d/1xLfSQUGDl8ezNNbUkpuHOYvSpTyxVhCs/view?usp=sharing

## Conclusions
- We build total 3 models, first model does not try to fix class imbalance or any other data issue and showed 93.9% accuracy on train while 84.4 on validation data.
- Second model uses Data Augmentation Strategy to make the model more robust, but it ended up creating highly underfitted model 
- In final model, Data aurmentation is being used to add 1000 images for each class to handle class imbalance, along with 40 epochs, and this model performed well. Peformance matrics are given below:

* Model showing great improvement in accuracy as accuracy for Train data is 94.31 while for test data it's 91.99 %.
* Loss graph also showing some fluctuation as begining but later it's setteled for 2-3% range on both validation and train data.
* Model benefitted largaly with Aurgumentation strategy as it helped fix class imbalance problem.

## Technologies Used
* **Library 1:** tensorflow version: 2.19.0
* **Library 2:** keras version: 3.8.0
* **Library 3:** numpy version: 2.0.2
* **Library 4:** pandas version: 2.2.2
* **Library 5:** PIL version: 11.1.0
* **Library 6:** Augmentor version: 0.2.12


## Acknowledgements
- Thanks for all the upgrad faculty members for all the necessory learning.


## Contact
Created by: Dinesh Namdev