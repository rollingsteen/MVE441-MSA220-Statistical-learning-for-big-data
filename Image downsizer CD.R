library(ElemStatLearn)
library(nsprcomp)
library(imager)

#--- Cat and dogs ---
catdogdata<-as.matrix(read.table("catdogdata.txt")) # np.loadtxt in python
Labels<-rep(0,198)
Labels[100:198]<-1
#
rotateM <- function(x) t(apply(x, 2, rev)) # the images are raster scans. Here, I just resort them for the
# default image command in R to plot them with the right orientation.
#
library(repr)
options(repr.plot.width=12, repr.plot.height=6)
#
set.seed(1000012)
ssc<-sample(seq(1,198)[Labels==0],2,replace=F)
ssd<-sample(seq(1,198)[Labels==1],2,replace=F)
par(mfrow=c(1,2))


downsample_block <- function(img_vector, original_size = 64, target_size = 32) {
  factor <- original_size / target_size
  mat <- matrix(img_vector, nrow = original_size, ncol = original_size)
  result <- matrix(0, nrow = target_size, ncol = target_size)
  
  for (i in 1:target_size) {
    for (j in 1:target_size) {
      block <- mat[((i-1)*factor+1):(i*factor), ((j-1)*factor+1):(j*factor)]
      result[i, j] <- mean(block)
    }
  }
  as.vector(result)
}

resize_images_manual <- function(image_matrix, original_size = 64, target_size = 32) {
  resized <- t(apply(image_matrix, 1, downsample_block, original_size, target_size))
  return(resized)
}


par(mfrow=c(1,2))
image(seq(1,32),seq(1,32),rotateM(matrix(catdogdata[ssc[1],],32,32)),col=gray.colors(256),xlab="",ylab="")
image(seq(1,32),seq(1,32),rotateM(matrix(catdogdata[ssc[2],],32,32)),col=gray.colors(256),xlab="",ylab="")
image(seq(1,32),seq(1,32),rotateM(matrix(catdogdata[ssd[1],],32,32)),col=gray.colors(256),xlab="",ylab="")
image(seq(1,32),seq(1,32),rotateM(matrix(catdogdata[ssd[2],],32,32)),col=gray.colors(256),xlab="",ylab="")



library(MASS)
library(caret)
library(tidyverse)

catdogdata<-as.matrix(read.table("catdogdata.txt")) # np.loadtxt in python
catdogdata <- resize_images_manual(catdogdata, original_size = 64, target_size = 32)
Labels<-rep(0,198)
Labels[100:198]<-1
data<-catdogdata
data <- as.data.frame(scale(data))
labels<-Labels

nzv <- nearZeroVar(data)
if (length(nzv) > 0) {
  data <- data[, -nzv]
  cat(length(nzv), "near-zero variance features removed.\n")
}

n_sig <- c()
test_error <- c()
size <-c(3072, 2048, 1024, 512, 256, 128, 64, 32, 16, 8, 4)
selected_feature_list <- list()

ldaFuncs_mod <- ldaFuncs
ldaFuncs_mod$selectSize <- pickSizeTolerance

for(ii in 1:5){
  cat("outer fold", ii, '\n')
  trainvalidationIndex <- createDataPartition(labels, p=0.8, list=F, times=1)
  data.test <- data[-trainvalidationIndex,]
  labels.test <- labels[-trainvalidationIndex]
  data.trainvalidation <- data[trainvalidationIndex,]
  labels.trainvalidation <- labels[trainvalidationIndex]
  
  control <- rfeControl(functions = ldaFuncs, method = "cv", number = 5, 
                        rerank = TRUE, verbose = FALSE)
  
  rfe_result <- rfe(
    x = data.trainvalidation,
    y = as.factor(labels.trainvalidation),
    sizes = size,
    rfeControl = control
  )
  
  selected_features <- predictors(rfe_result)
  selected_feature_list[ii] <- selected_features
  n_sig[ii] <- length(selected_features)
  
  data.train.dimred <- data.trainvalidation[, selected_features]
  data.test.dimred <- data.test[, selected_features]
  
  fit <- lda(labels.trainvalidation ~ ., data = as.data.frame(data.train.dimred))
  pred <- predict(fit, newdata = as.data.frame(data.test.dimred))
  
  test_error[ii] <- sum(pred$class != labels.test) / length(labels.test)
}


selected_features <- selected_feature_list[[2]]

# Convert "V123" to numeric 123
feature_indices <- as.numeric(gsub("V", "", selected_features))

# Initialize feature map
image_size <- 32
feature_map <- matrix(0, nrow = image_size, ncol = image_size)

# Mark selected features
for (i in feature_indices) {
  row <- ((i - 1) %% image_size) + 1
  col <- ((i - 1) %/% image_size) + 1
  feature_map[row, col] <- 1
}

# Plot feature map
image(t(apply(feature_map, 2, rev)), col = c("white", "black"),
      axes = FALSE, main = "Selected Features (RFE) - Fold 2")



