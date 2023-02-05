# Librerías
library(MASS)
library(statmod)

datos=read.csv('ExamenP2.csv')

# LIGA IDENTIDAD
model_id<-glm(Y~X, family=poisson(link='identity'),data=datos)
summary(model_id)
anova(model_id,test = 'Chisq')

# Gráfica
new_casos <- predict(model_id,se.fit=TRUE)
mu_model_identity <- new_casos$fit
plot( datos$Y~datos$X, las=1, main="Modelo Liga Identidad")
lines( mu_model_identity ~ datos$X, lwd=2 )

# LIGA CANONICA (LOG)
model_log<-glm(Y~X, family=poisson(link='log'),data=datos)
summary(model_log)
anova(model_log,test = 'Chisq')

# Gráfica
new_casos <- predict(model_log,se.fit=TRUE)
mu_model_log <- new_casos$fit
plot( datos$Y~datos$X, las=1, main="Modelo Liga Logaritmo")
lines( exp(mu_model_log) ~ datos$X, lwd=2 )


