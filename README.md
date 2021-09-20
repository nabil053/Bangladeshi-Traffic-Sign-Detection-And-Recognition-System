# Bangladeshi-Traffic-Sign-Detection-And-Recognition-System

A system for detecting and recognizing one of the six common kinds of Bangladeshi traffic signs from input video using and SVM model. The classification is done based on the HOG property.

The dataset used for training the model consists of 3488 Bangladeshi traffic sign images divided into six classes of Bangladeshi traffic signs, labelled 0-6, standing for 'do not turn left' (788), 'no entry' (684), 'no horn' (412), 'no parking' (480), 'no rickshaw' (524) and 'speed limit 30 mph' (600) respectively. All the images are stored in the 'dataset' folder.

The SVM model generated is the standard SVM model utilizing a radial basis function kernel. The images are classified based on the Histogram of Oriented Gradient property. The model is generated using the 'classification.py' file. The generated model is stored in 'data_svm.dat' file.

The 'main.py' file is where the system is run. Here, a video is provided as input, frame by frame. After preprocessing each frame, the largest traffic sign-like object is detected and fed to the classifier to recognize its type. The resultant video with detection is written to 'output.avi'. The videos titled 'vid_1' to 'vid_6' are used to test the system's effectiveness.
