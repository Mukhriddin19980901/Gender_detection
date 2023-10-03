##  **Gender detection**

<img src="https://github.com/Mukhriddin19980901/Gender_detection/blob/main/pngfile/face_analysis_camera_selector-02.jpg?raw=true" width="600" height="500" />

## 1.Introduction

- The aim of this project is to develop an algorithm that can accurately classify an individual's gender (male,female) based on facial features extracted from images.Creating real-time processing of facial images to provide instantaneous gender classification, making the system applicable to live video feeds or applications that require immediate responses.

<img src="https://github.com/Mukhriddin19980901/Gender_detection/blob/main/pngfile/realt-face-detection.gif?raw=true" width="600" height="500" />


- For this project I have classified overall 56,000 images to identify a gender of people from different races.The images sizes are different.


## 2.Code description

The code is divided into 4 main parts:

**introduction to data & visualization , preprocessing datas , building a model,training & testing**

- Intially  all images are uploaded and learnt the types(images are all jpg format in female and male seperate files ), size ,proportion balance and qulaity.

<img src="https://github.com/Mukhriddin19980901/Gender_detection/blob/main/pngfile/gender_dtc.png?raw=true" width="600" height="500" />

- Datas are standardized into **96** to **96** and **RGB** format and labels are encoded through one hot encoding.

<img src="https://github.com/Mukhriddin19980901/Gender_detection/blob/main/pngfile/genders.png?raw=true" width="500" height="300" />

- The model is built through  convolutional neural networks using [**Tensorflow**](https://en.wikipedia.org/wiki/TensorFlow)  
