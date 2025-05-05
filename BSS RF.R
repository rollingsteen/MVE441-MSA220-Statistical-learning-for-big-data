library(ElemStatLearn)
library(nsprcomp)
library(caret)
library(doParallel)
library(repr)

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
image(seq(1,64),seq(1,64),rotateM(matrix(catdogdata[ssc[1],],64,64)),col=gray.colors(256),xlab="",ylab="")
image(seq(1,64),seq(1,64),rotateM(matrix(catdogdata[ssc[2],],64,64)),col=gray.colors(256),xlab="",ylab="")
image(seq(1,64),seq(1,64),rotateM(matrix(catdogdata[ssd[1],],64,64)),col=gray.colors(256),xlab="",ylab="")
image(seq(1,64),seq(1,64),rotateM(matrix(catdogdata[ssd[2],],64,64)),col=gray.colors(256),xlab="",ylab="")

catdog_df <- data.frame(catdogdata)
catdog_df$Label <- as.factor(Labels)


ctrl_rf <- rfeControl(
  functions    = rfFuncs,
  method       = "cv",
  number       = 10,
  verbose      = TRUE,
  returnResamp = "final"
)

predictors <- as.data.frame(catdog_df[, -ncol(catdog_df)])
response <- catdog_df$Label

set.seed(2025)
rfe_rf <- rfe(
  x          = predictors,
  y          = response,
  sizes      = c(4096, 2048, 1024, 512, 256, 128, 64),
  rfeControl = ctrl_rf,
  method     = "rf",                       
  tuneGrid   = expand.grid(mtry = c(16, 64, 128))
)

print(rfe_rf)
plot(rfe_rf)



selected_vars <- rfe_rf$optVariables
selected_idx <- as.integer( gsub("[^0-9]", "", selected_vars) )

if(any(is.na(selected_idx))) {
  warning("Some variable names could not be converted to indices:")
  print(selected_vars[is.na(selected_idx)])
}

mask <- matrix(0, nrow = 64, ncol = 64)
for(i in selected_idx){
  r <- ((i - 1) %% 64) + 1
  c <- ((i - 1) %/% 64) + 1
  mask[r, c] <- 1
}

par(mfrow = c(1,2))
image(1:64, 1:64, t(mask[64:1,]), col = c("white","black"),
      xlab="Column", ylab="Row", main=paste(length(selected_idx),"pixels"))
      
      