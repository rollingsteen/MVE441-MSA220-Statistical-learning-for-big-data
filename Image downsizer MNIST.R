library(ElemStatLearn)
library(nsprcomp)

data(zip.train)
set.seed(1000)
Numbers<-zip.train # a subset of the digits
jj<-seq(1,7291)[Numbers[,1]==9]
JJ<-t(apply(Numbers[jj,-1],1,rev)) # turn the 9s upside down
Numbers2<-rbind(Numbers[jj,],cbind(matrix(rep(-9,dim(JJ)[1]),dim(JJ)[1],1),JJ))
jj<-seq(1,7291)[Numbers[,1]==2]
JJ<-t(apply(Numbers[jj,-1],1,rev)) # turn the 2s upside down
Numbers2<-rbind(Numbers2,Numbers[jj,],cbind(matrix(rep(-2,dim(JJ)[1]),dim(JJ)[1],1),JJ))
jj<-seq(1,7291)[Numbers[,1]==5]
Numbers2<-rbind(Numbers2,Numbers[jj,])
jj<-seq(1,7291)[Numbers[,1]==6]
Numbers2<-rbind(Numbers2,Numbers[jj,])
jj<-seq(1,7291)[Numbers[,1]==8]
Numbers2<-rbind(Numbers2,Numbers[jj,])
jj<-seq(1,7291)[Numbers[,1]==0]
Numbers2<-rbind(Numbers2,Numbers[jj,])
jj<-seq(1,7291)[Numbers[,1]==1]
Numbers2<-rbind(Numbers2,Numbers[jj,])
#
ii<-sample(seq(1,dim(Numbers2)[1]),2000) # a subset to use for the project
Numbers<-Numbers2[ii,]
labs<-Numbers[,1]
table(Numbers[,1])

par(mfrow=c(3,3)) # The digits can look quite different - run this a couple of times to see.
labu<-unique(labs)
for (zz in (seq(1,9))) {
  iz<-sample(seq(1,dim(Numbers)[1])[Numbers[,1]==labu[zz]],1)
  image(t(matrix(as.numeric(Numbers[iz,-1]),16,16,byrow=T))[,16:1], col = gray.colors(33))
}

downsample_block <- function(img_vector, original_size = 16, target_size = 8) {
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

resize_images_manual <- function(image_matrix, original_size, target_size) {
  resized <- t(apply(image_matrix, 1, downsample_block, original_size, target_size))
  return(resized)
}

digits_data <- as.matrix(Numbers[, -1])
digits_labels <- Numbers[, 1]           

digits_8x8 <- resize_images_manual(digits_data, original_size = 16, target_size = 8)

par(mfrow=c(3,3))
idx <- sample(1:nrow(digits_8x8), 9)
for (i in idx) {
  image(seq(1,8), seq(1,8), rotateM(matrix(digits_8x8[i,], 8, 8)), 
        col = gray.colors(256), xlab = paste("Label:", digits_labels[i]), ylab = "")
}

library(MASS)
library(caret)
library(tidyverse)

digits_data <- as.matrix(Numbers[, -1])
digits_labels <- Numbers[, 1]

data <- as.data.frame(scale(digits_data))
labels <- digits_labels


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
  selected_feature_list[[ii]] <- selected_features
  n_sig[ii] <- length(selected_features)
  
  data.train.dimred <- data.trainvalidation[, selected_features]
  data.test.dimred <- data.test[, selected_features]
  
  fit <- lda(labels.trainvalidation ~ ., data = as.data.frame(data.train.dimred))
  pred <- predict(fit, newdata = as.data.frame(data.test.dimred))
  
  test_error[ii] <- sum(pred$class != labels.test) / length(labels.test)
}


selected_features <- selected_feature_list[[2]]

feature_indices <- as.numeric(gsub("V", "", selected_features))

image_size <- 16
feature_map <- matrix(0, nrow = image_size, ncol = image_size)

for (i in feature_indices) {
  row <- ((i - 1) %% image_size) + 1
  col <- ((i - 1) %/% image_size) + 1
  feature_map[row, col] <- 1
}

image(t(apply(feature_map, 2, rev)), col = c("white", "black"),
      axes = FALSE, main = "Selected Features (RFE) - Fold 2")

