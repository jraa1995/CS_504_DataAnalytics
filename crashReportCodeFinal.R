library(readr)
library(stargazer)
library(jtools)
library(plm)

dat2 <- read.csv("C:/Users/annaf/Downloads/Final_Crash - crashreport.csv")

#data exploration
head(dat)
table(dat$light)
table(dat$driverDistractedBy)
table(dat$vehicleMake)
table(dat$crashDateTime)

table(dat$injurySeverity)

#The purpose of this model is to predict whether there are certain factors
#that affect whether someone is seriously injured in a car crash. 
#We have to create binary variables.

#Dependent variable = Injury
dat2$injured <- ifelse(grepl("FATAL INJURY|SUSPECTED SERIOUS INJURY|POSSIBLE INJURY|SUSPECTED MINOR INJURY",
                             dat2$injurySeverity),1,0)
dat2$uninjured <- ifelse(dat2$injurySeverity== "NO APPARENT INJURY",1,0)

table(dat2$injured)
table(dat2$uninjured)
table(dat$trafficControl)

#Independent variable 
#First: Do speed limits impact the likelihood of injury?
#I think this is a particularly important variable in regards to policy decisions.
table(dat2$speedLimit)


dat2$speed0<- ifelse(dat2$speedLimit == 0,1,0) #if speedLimit=0, assign 1. else, assign 0
dat2$speed5<- ifelse(dat2$speedLimit == 5,1,0)
dat2$speed10<- ifelse(dat2$speedLimit == 10,1,0)
dat2$speed15<- ifelse(dat2$speedLimit == 15,1,0)
dat2$speed20<- ifelse(dat2$speedLimit == 20,1,0)
dat2$speed25<- ifelse(dat2$speedLimit == 25,1,0)
dat2$speed30<- ifelse(dat2$speedLimit == 30,1,0)
dat2$speed35<- ifelse(dat2$speedLimit == 35,1,0)
dat2$speed40<- ifelse(dat2$speedLimit == 40,1,0)
dat2$speed45<- ifelse(dat2$speedLimit == 45,1,0)
dat2$speed50<- ifelse(dat2$speedLimit == 50,1,0)
dat2$speed55<- ifelse(dat2$speedLimit == 55,1,0)
dat2$speed60<- ifelse(dat2$speedLimit == 60,1,0)
dat2$speed65<- ifelse(dat2$speedLimit == 65,1,0)
dat2$speed70<- ifelse(dat2$speedLimit == 70,1,0)
dat2$speed75<- ifelse(dat2$speedLimit == 75,1,0)

#Second: Are areas with traffic control more prone to accidents with injuries?
dat2$trafficControl[dat2$trafficControl ==""] <- "UNKNOWN"
dat2$yestrafficControl <- ifelse(grepl("FLASHING TRAFFIC SIGNAL|OTHER|PERSON|RAILWAY CROSSING DEVICE|SCHOOL ZONE SIGN DEVICE|STOP SIGN|TRAFFIC SIGNAL|WARNISSANNG SIGN|YIELD SIGN", dat2$trafficControl),1,0)


dat2$noTrafficControl <- ifelse(dat2$trafficControl == "NO CONTROLS",1,0)
dat2$trafficControlUnknown <- ifelse(dat2$trafficControl == "UNKNOWN",1,0)

#validating that values add up
table(dat2$yestrafficControl)
table(dat2$noTrafficControl)
table(dat2$trafficControlUnknown)

#subsetting only for values where traffic control is known
subdat = subset(dat2, trafficControl != "UNKNOWN")

#Here is the model:
lpm3= glm(injured~ yestrafficControl+ speed0+
            speed5+ speed10+speed15+speed20+speed25+
            speed30+speed35+speed45+speed50+speed55+
            speed60+speed65+speed70+speed75,
          data= subdat, family= binomial(link='probit'))

summary(lpm3, digits=4)

#marginal effects
subdat$phi_z= dnorm(subdat$z_hat,0,1)
mean(subdat$phi_z)

probitScalar <- mean(dnorm(predict(lpm3, type= "link")))
print(probitScalar)

probitScalar * coef(lpm3) #this gives each variables marginal effects


#checking predictions
subdat$prob_injured = predict(lpm3, type="response")
subdat$z_hat = lpm3$linear.predictors

max(subdat$prob_injured) #max probability of injury is 0.2836

subdat$injured_hat= ifelse(subdat$prob_injured >= 0.24,1,0) #assign 1 to injured_hat
#if the predicted value is greater or equal to 0.24. else, assign 0

subdat$correct= ifelse(subdat$injured==subdat$injured_hat, 1,0)
forstats= c("injured", "injured_hat", "correct")
stargazer(subdat[forstats], type="text") #table with predicted values