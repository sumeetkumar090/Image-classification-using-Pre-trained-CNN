#Task 3- Sumeet Kumar - 5873137 - Sk521
#cran <- getOption("repos")
#cran["dmlc"] <- "https://s3-us-west-2.amazonaws.com/apache-mxnet/R/CRAN/"
#options(repos = cran)
#install.packages("mxnet",dependencies = T)
#install.packages('imager')
library(mxnet)
require(imager)
require(e1071)
library(tensorflow)
library(keras)
library(pROC)


setwd("/")

#Use the model loading function to load the model into R and preprocess the data
model = mx.model.load("Inception/Inception_BN", iteration=39)
mean.img = as.array(mx.nd.load("Inception/mean_224.nd")[["mean_img"]])

#Load and Preprocess the Image
im <- load.image(system.file("extdata/parrots.png", package="imager"))
plot(im)

#Before feeding the image to the deep network, we need to perform some preprocessing to make the image meet the deep network input requirements.
preproc.image <- function(im, mean.image) {
  # crop the image
  shape <- dim(im)
  if (shape[4] != 3){
    im <- add.color(im)
    shape <- dim(im)
  }
  short.edge <- min(shape[1:2])
  xx <- floor((shape[1] - short.edge) / 2)
  yy <- floor((shape[2] - short.edge) / 2)
  cropped <- crop.borders(im, xx, yy)
  # resize to 224 x 224, needed by input of the model.
  resized <- resize(cropped, 224, 224)
  # convert to array (x, y, channel)
  arr <- as.array(resized) * 255
  dim(arr) <- c(224, 224, 3)
  # subtract the mean
  normed <- arr - mean.img
  # Reshape to format needed by mxnet (width, height, channel, num)
  dim(normed) <- c(224, 224, 3, 1)
  return(normed)
}

# load caltech images directory - test and train dataset
train_imageData <- flow_images_from_directory("datasets/train", generator = image_data_generator(), target_size = c(224, 224))
test_imageData <- flow_images_from_directory("datasets/test", generator = image_data_generator(), target_size = c(224, 224))

#Extracting feature from training dataset
trainFeatures <- list()
y_train <- list()
i<-1
while(i<=length(train_imageData$filenames))
{
  # load image one by one and process it
  k <- paste("datasets/train/",as.list(train_imageData$filenames[i]),sep = "") 
  img <- load.image(k)
  normedDataset = preproc.image(img, mean.img)
  
  internals = model$symbol$get.internals()
  fea_symbol = internals[[match("global_pool_output", internals$outputs)]]
  
  trainModel <- list(symbol = fea_symbol,
                     arg.params = model$arg.params,
                     aux.params = model$aux.params)
  
  class(trainModel) <- "MXFeedForwardModel"
  
  trainFeatures[[i]] <- predict(trainModel, X = normedDataset, allow.extra.params = TRUE)
  
  print(i)
  i=i+1
}

d = do.call(rbind, trainFeatures)
#write the result to csv
write.csv(d,"train.csv")
#train the model using SVM
svmModel<-svm(x=d,as.factor(train_imageData$classes),scale=F, kernel="radial", gamma=0.001, cost=10)

#Extracting feature from testing dataset
testFeatures <- list()
i<-1
while(i<=length(test_imageData$filenames))
{
  # load image one by one and process it
  k <- paste("datasets/test/",as.list(test_imageData$filenames[i]),sep = "") 
  img <- load.image(k)
  normedDataset1 = preproc.image(img, mean.img)
  
  internals = model$symbol$get.internals()
  fea_symbol = internals[[match("global_pool_output", internals$outputs)]]
  
  testModel <- list(symbol = fea_symbol,
                    arg.params = model$arg.params,
                    aux.params = model$aux.params)
  
  class(testModel) <- "MXFeedForwardModel"
  
  testFeatures[[i]] <- predict(testModel, X = normedDataset1, allow.extra.params = TRUE)
  
  print(i)
  i=i+1
}

d1 = do.call(rbind, testFeatures)
#write the result to csv
write.csv(d1,"test.csv")
#Predict the test result
svmPredictions <- predict(svmModel,d1)
#predictions

mean(svmPredictions==test_imageData$classes)

temp <- table("Actual" = test_imageData$classes, "Predictions" = svmPredictions)

#Draw confusion matrix
cm<-confusionMatrix(temp)
cm
#write CM result
write.csv(as.matrix(cm),"conf.csv")

#Plot multiclass ROC - not working in R
#rs <- roc.multi[['rocs']]
#plot.roc(rs[[1]])
#sapply(2:length(rs),function(i) lines.roc(rs[[i]],col=i))


# This code is just for reference - process using feature extracted csv files.
# The Commented code for only classification using SVM from the train and test feature extracted file.
# If required the feature extracted csv files can also be provided but not attached with the assignment.
#setwd("/")

#training set
#training_set <- flow_images_from_directory("dataset/train", generator = image_data_generator(), target_size = c(224, 224))
#testing set
#testing_set <- flow_images_from_directory("dataset/test", generator = image_data_generator(), target_size = c(224, 224))

#Read train and test csv.
#trainData <- read.csv(file= "train.csv",  head=TRUE, sep=",")
#trainLabel <- as.factor(training_set$classes)
#testData <- read.csv(file= "test.csv",  head=TRUE, sep=",")
#testLabel <- as.factor(testing_set$classes)

#model
#model  <- svm(x=as.matrix(trainData),y=trainLabel, kernel="radial", gamma=0.001, cost=10)

#Prediction
#prediction <- predict(model, testData)

#temp <- table("Actual" = testLabel, "Predictions" = prediction)

#Draw confusion matrix
#cm<-confusionMatrix(temp)
#cm$overall