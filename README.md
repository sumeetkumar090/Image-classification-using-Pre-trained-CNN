# Image-classification-using-Pre-trained-CNN

1.	Firstly let us download the pre-trained batch inception network from the below link:
http://data.mxnet.io/mxnet/data/Inception.zip
2.	Now load the mxnet and imager package to load and preprocess the images in R.
3.	Load the pre-trained model and the mean image which is used for pre-processing.
4.	Use the below code to load an image from the system or from the imager package:
#im <- load.image(system.file("extdata/boat.png", package="imager"))
im <- load.image("task3/train/train/002.american-flag/002_0002.jpg")
plot(im)
5.	Now before feeding the image to deep network, we need to perform some preprocessing to make the image meet the deep network input requirements.  Preprocessing includes cropping and subtracting the mean. Because MXNet is deeply integrated with R, we can do all the processing in an R function.
6.	Now we will use the defined preprocessing function to get the normalized image.
7.	We will now use the predict function to get the probability over classes.
8.	Use the max.col on the transpose of prob to get the class index.
9.	Once it is done read the name of the classes from the synset.txt file.
10.	And at last print the predicted name for the image
synsets <- readLines("task3/Inception/synset.txt")
print(paste0("Predicted Top-class: ", synsets  [[max.idx]]))

