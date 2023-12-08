AVMT <- read.csv("./MS/CS-504/Team Project/Street-Count.csv")

test <- t.test(AVMT$FREQUENCY, AVMT$AVMT )

test

