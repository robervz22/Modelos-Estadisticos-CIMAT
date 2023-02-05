# Librerías
library(MASS)
library(statmod)

datos=read.csv('P5Plac.csv')
attach(datos)

plot(Numero,Edad,pch=16,xlab='Edad',ylab='Num Polipos',main='Grupo Placebo')

# Poisson
out1=glm(Numero~Edad,family=poisson)
summary(out1)

# Quasi-Poisson
out2=glm(Numero~Edad,family = quasipoisson)
summary(out2)
anova(out2,test = 'Chisq')

# Binomial Negativo
out3=glm.nb(Numero~Edad)
summary(out3)
out3=glm.convert(out3)
anova(out3,test='Chisq')


newEdad <- seq( min(datos$Edad), max(datos$Edad), length=100)
newNum.qp <- predict(out2, newdata=data.frame(Edad=newEdad),se.fit=TRUE)
newNum.nb <- predict(out3, newdata=data.frame(Edad=newEdad),se.fit=TRUE,
                     dispersion=1)
tstar <- qt(0.975, df=df.residual(out2) ) # For a 95% CI
ME.qp <- tstar * newNum.qp$se.fit; ME.nb <- tstar * newNum.nb$se.fit
mu.qp <- newNum.qp$fit; mu.nb <- newNum.nb$fit

par( mfrow=c(1, 2))
plot( Numero~Edad, las=1, main="Valores ajustados")
lines( exp(mu.qp) ~ newEdad, lwd=2 )
lines( exp(mu.nb) ~ newEdad, lwd=2, lty=2 );
legend("topleft", lty=1:2, legend=c("QP", "NB") )
#
plot( Numero~Edad, las=1, main="Intervalos de confianza")
ci.lo <- exp(mu.qp - ME.qp); ci.hi <- exp(mu.qp + ME.qp)
lines( ci.lo ~ newEdad, lwd=2); lines( ci.hi ~ newEdad, lwd=2)
ci.lo <- exp(mu.nb - ME.nb); ci.hi <- exp(mu.nb + ME.nb)
lines( ci.lo ~ newEdad, lwd=2, lty=2)
lines( ci.hi ~ newEdad, lwd=2, lty=2)
#legend("topleft", lty=1:2, legend=c("QP", "NB") )
