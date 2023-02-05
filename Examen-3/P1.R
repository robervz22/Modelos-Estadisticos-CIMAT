# Librerías
library(MASS)
library(statmod)

datos=read.csv('ExamenP1.csv')
# MODELO 1
model1<-glm(cbind(leucemia,total-leucemia)~dosis, family=binomial,data=datos)
summary(model1)
anova(model1,test = 'Chisq')

# Grafica
newDosis <- seq( min(datos$dosis), max(datos$dosis), length=6)
newleucemia <- predict(model1, newdata=data.frame(dosis=newDosis),se.fit=TRUE)
mu_model1 <- newleucemia$fit
plot( datos$leucemia~datos$dosis, las=1, main="Modelo 1")
model1_prop=exp(mu_model1)/(1+exp(mu_model1))
lines( datos$total*model1_prop ~ newDosis, lwd=2 )

# MODELO 2
model12<-glm(cbind(leucemia,total-leucemia)~dosis+dosis2, family=binomial,data=datos)
summary(model12)
anova(model12,test = 'Chisq')

# Grafica
newDosis <- seq( min(datos$dosis), max(datos$dosis), length=6)
newleucemia <- predict(model12, newdata=data.frame(dosis=newDosis,dosis2=newDosis^2),se.fit=TRUE)
mu_model12 <- newleucemia$fit
plot( datos$leucemia~datos$dosis, las=1, main="Modelo 2")
model12_prop=exp(mu_model12)/(1+exp(mu_model12))
lines( datos$total*model12_prop ~ newDosis, lwd=2 )


# MODELO 3
model123<-glm(cbind(leucemia,total-leucemia)~dosis+dosis2+dosis3, family=binomial,data=datos)
summary(model123)
anova(model123,test = 'Chisq')

# Grafica
newDosis <- seq( min(datos$dosis), max(datos$dosis), length=6)
newleucemia <- predict(model123, newdata=data.frame(dosis=newDosis,dosis2=newDosis^2,dosis3=newDosis^3),se.fit=TRUE)
mu_model123 <- newleucemia$fit
plot( datos$leucemia~datos$dosis, las=1, main="Modelo 3")
model123_prop=exp(mu_model123)/(1+exp(mu_model123))
lines( datos$total*model123_prop ~ newDosis, lwd=2 )


# DERIVADOS

# 2 variables explicativas
model13<-glm(cbind(leucemia,total-leucemia)~dosis+dosis3, family=binomial,data=datos)
summary(model13)
anova(model13,test = 'Chisq')

model23<-glm(cbind(leucemia,total-leucemia)~dosis2+dosis3, family=binomial,data=datos)
summary(model23)
anova(model23,test = 'Chisq')

# 1 variable explicativa
model2<-glm(cbind(leucemia,total-leucemia)~dosis2, family=binomial,data=datos)
summary(model2)
anova(model2,test = 'Chisq')

model3<-glm(cbind(leucemia,total-leucemia)~dosis3, family=binomial,data=datos)
summary(model3)
anova(model3,test = 'Chisq')

# Grafica
newDosis <- seq( min(datos$dosis), max(datos$dosis), length=6)
newleucemia <- predict(model3, newdata=data.frame(dosis3=newDosis^3),se.fit=TRUE)
mu_model3 <- newleucemia$fit
plot( datos$leucemia~datos$dosis, las=1, main="Modelo con variable Dosis 3")
model3_prop=exp(mu_model3)/(1+exp(mu_model3))
lines( datos$total*model3_prop ~ newDosis, lwd=2 )

