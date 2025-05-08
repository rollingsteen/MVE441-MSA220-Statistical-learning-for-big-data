library(caret)
library(MASS)


#Numbers data
library(ElemStatLearn)
library(nsprcomp)

data(zip.train)
set.seed(1000)
Numbers<-zip.train # a subset of the digits
#
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
ii<-sample(seq(1,dim(Numbers2)[1]),200) # a subset to use for the project
Numbers<-Numbers2[ii,]
data <- Numbers[,-1]
labels <- Numbers[,1]

#Cat dog data
catdogdata<-as.matrix(read.table("catdogdata.txt")) 
Labels<-rep(0,198)
Labels[100:198]<-1
#data<-catdogdata
#labels<-Labels


#--- Cross Validation ---
F_test <- function(x,y){
  mod<-aov(x~as.factor(y))
  p_vals<-anova(mod)[1,5]
  return(p_vals)
}

n_sig <- c()
alpha <- c(0.5,0.1,0.05,0.01)
test_error <-c()
opt_alpha<-c()
img_p_sig<-c()

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
  opt_alpha[ii] <- which(errors==min(errors), arr.ind=T)
  sig_p.val <- seq(1,dim(data.train)[2])[p.adj<opt_alpha[ii]]
  data.trainvalidation.dimred <- data.trainvalidation[,sig_p.val]
  data.test.dimred <- data.test[,sig_p.val]
  n_sig[kk] <- length(sig_p.val)
  img_p_sig[ii]<-sig_p.val
  
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
  test_error[ii] <- sum(pred != labels.test) / length(labels.test)
  
  
}
boxplot(test_error,ylab="error rate",main="error rate for rf with Ftest")
raster_vec <- rep(1, 16 * 16)

# Set some random pixel indices to black (0)
raster_vec[sig_p.val] <- 0

# Convert the vector to a 64x64 matrix
img <- matrix(raster_vec, nrow = 16, ncol = 16, byrow = TRUE)

# Display the image with proper orientation
image(
  1:16, 1:16, t(apply(img, 2, rev)),  # Rotate for correct orientation
  col = gray.colors(2, start = 0, end = 1),
  xlab = "", ylab = "", axes = FALSE, asp = 1)

