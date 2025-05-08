
library(imager)
library(caret)
library(MASS)


#Numbers data
library(ElemStatLearn)
library(nsprcomp)

downsample_to <- function(image_vector, original_size = 64, target_size = 32, blur_sigma = 1, return_matrix = FALSE) {
  # Case 1: No downsampling needed
  if (original_size == target_size) {
    if (return_matrix) {
      return(matrix(image_vector, nrow = target_size, byrow = TRUE))
    } else {
      return(image_vector)
    }
  }
  
  # Case 2: Downsample
  img_matrix <- matrix(image_vector, nrow = original_size, byrow = TRUE)
  img_cimg <- as.cimg(img_matrix)
  img_cimg <- isoblur(img_cimg, sigma = blur_sigma)
  scale_factor <- target_size / original_size
  img_resized <- imresize(img_cimg, scale = scale_factor)
  
  if (return_matrix) {
    return(as.matrix(img_resized))
  } else {
    return(as.numeric(img_resized))
  }
  testerrors <- matrix(NA, nrow = 4, ncol = 5)
  sizes<-c(64, 32, 16, 8)
  
  for (size in 1:length(sizes)) {
    catdogdata<-as.matrix(read.table("catdogdata.txt")) 
    catdog_down <- t(apply(catdogdata, 1, downsample_to, target_size = sizes[size]))
    Labels<-rep(0,198)
    Labels[100:198]<-1
    data<-catdog_down
    labels<-Labels
    
    
    #--- Cross Validation ---
    F_test <- function(x,y){
      mod<-aov(x~as.factor(y))
      p_vals<-anova(mod)[1,5]
      return(p_vals)
    }
    
    n_sig <- c()
    alpha <- c(0.5,0.1,0.05,0.01)
    #test_error <-c()
    opt_alpha<-c()
    
    for (ii in 1:5){
      trainvalidationIndex <- createDataPartition(labels, p=0.8, list=F, times=1)
      data.test <- data[-trainvalidationIndex,]
      labels.test <- labels[-trainvalidationIndex]
      data.trainvalidation <- data[trainvalidationIndex,]
      labels.trainvalidation <- labels[trainvalidationIndex]
      
      errors <- rep(0,length(alpha))
      for(fold in 1:5){
        foldIndex <- createDataPartition(labels.trainvalidation, p=0.8, list=F, times=1)
        data.train <- data.trainvalidation[foldIndex,]
        labels.train <- labels.trainvalidation[foldIndex]
        data.validation <- data.trainvalidation[-foldIndex,]
        labels.validation <- labels.trainvalidation[-foldIndex]  
        
        Ftest.res <- apply(data.train,2,F_test,y=as.factor(labels.train))
        p.adj<-p.adjust(Ftest.res,"BH")
        
        for(kk in 1:length(alpha)){
          sig_p.val <- seq(1,dim(data.train)[2])[p.adj<alpha[kk]]
          data.train.dimred <- data.train[,sig_p.val]
          data.validation.dimred <- data.validation[,sig_p.val]
          n_sig[kk] <- length(sig_p.val)
          
          colnames(data.train.dimred) <- paste0("V", seq_len(ncol(data.train.dimred)))
          colnames(data.validation.dimred) <- paste0("V", seq_len(ncol(data.validation.dimred)))
          
          
          fit <- train(
            x = data.train.dimred,
            y = as.factor(labels.train),
            method = "rf",
            trControl = trainControl(method = "none"),
            tuneGrid = data.frame(mtry = floor(sqrt(ncol(data.train.dimred)))) # or set a fixed mtry
          )
          pred <- predict(fit, newdata = data.validation.dimred)
          errors[kk] <- errors[kk] + sum(pred != labels.validation) 
        }
      }    
      opt_alpha[ii] <- which(errors==min(errors))[1]
      sig_p.val <- seq(1,dim(data.train)[2])[p.adj<alpha[opt_alpha[ii]]]
      data.trainvalidation.dimred <- data.trainvalidation[,sig_p.val]
      data.test.dimred <- data.test[,sig_p.val]
      n_sig[kk] <- length(sig_p.val)
      
      colnames(data.trainvalidation.dimred) <- paste0("V", seq_len(ncol(data.trainvalidation.dimred)))
      colnames(data.test.dimred) <- paste0("V", seq_len(ncol(data.test.dimred)))
      
      fit <- train(
        x = data.trainvalidation.dimred,
        y = as.factor(labels.trainvalidation),
        method = "rf",
        trControl = trainControl(method = "none"),
        tuneGrid = data.frame(mtry = floor(sqrt(ncol(data.trainvalidation.dimred))))
      )
      pred <- predict(fit, newdata = data.test.dimred)
      testerrors[size,ii] <- sum(pred != labels.test) / length(labels.test)
      print(sizes[size])
      
    }
    
  }
}
boxplot(as.data.frame(t(testerrors)), 
        names = paste("Row", 1:4), 
        main = "Boxplot of Each Row",
        ylab = "Error Rate")
