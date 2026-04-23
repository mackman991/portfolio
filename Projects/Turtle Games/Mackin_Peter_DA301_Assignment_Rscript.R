##################################################

# RUN Python script first

##################################################

# The whole tidyverse package
library(tidyverse)
# Import and read CSV file.
library(readr)
# Data wranglin
library(dplyr)
# Import the psych package.
library(psych)
# Data wrangling
library(tidyr)
# Create statistical summaries.
library(skimr)
# Create a report as an HTML file.
library(DataExplorer)
library(factoextra)

# Read file called turtle_clean2
tg <- read.csv(file.choose(), header = TRUE)

# Assess data
head(tg)
tail(tg)
dim(tg)
summary(tg)
glimpse(tg)
View(tg)

# Remove columns
tg2 <- select(tg, -review, -summary, -education)
head(tg2)

typeof(tg2$product)
tg2$product <- as.factor(tg$product)

DataExplorer::create_report(tg2)

##################################################

# Group age and income group
tg2$age_group <- cut(tg2$age,
                     breaks = c(0, 20, 35, 50, Inf),
                     labels = c("0–20", "21–35", "36–50", "50+"),
                     right = FALSE)

tg2$income_group <- cut(tg2$income,
                        breaks = c(0,30,50,75,100, Inf),
                        labels = c("Very Low", "Low","Middle","High", "very High"),
                        right = FALSE)


# Create a scatterplot to view the data set.
ggplot(tg2, aes(x=spend_score,
                  y=income,
                  colour = age_group)) +
  geom_point()


ggplot(tg2, aes(x=spend_score,
                y=loyalty_points,
                colour = income_group)) +
  geom_point()

#################################################

tg2$log_loyalty_points <- log(tg2$loyalty_points)

# Create a new object and specify variables.
model_tg = lm(log_loyalty_points~income+spend_score, data=tg2)

# Print the summary statistics and plot model.
summary(model_tg)
plot(model_tg)

plot(model_tg$fitted.values, model_tg$residuals,
     xlab = "Fitted values",
     ylab = "Residuals",
     main = "Residuals vs Fitted")

# QQ Plot
qqnorm(model_tg$residuals)
qqline(model_tg$residuals, col = "blue")

# Create a new object and specify variables.
model2_tg = lm(log_loyalty_points~income+spend_score+age, data=tg2)

# Print the summary statistics and plot model.
summary(model2_tg)
plot(model2_tg)

plot(model2_tg$fitted.values, model2_tg$residuals,
     xlab = "Fitted values",
     ylab = "Residuals",
     main = "Residuals vs Fitted")

# QQ Plot
qqnorm(model2_tg$residuals)
qqline(model2_tg$residuals, col = "blue")

################################################################

# Customer Segments

# Select features
tg2_cluster <- tg2 %>% select(income, spend_score)

# Standardise features
tg2_scaled <- scale(tg2_cluster)

# Elbow Method to determine optimal number of clusters
set.seed(42)
fviz_nbclust(tg2_scaled, kmeans, method = "wss") +
  geom_vline(xintercept = 0, linetype = 2) +
  labs(title = "Elbow Method")

# Fit K-Means model (K = 5)
set.seed(42)
kmeans_model <- kmeans(tg2_scaled, centers = 5, nstart = 25)

# Add cluster labels to original data
tg2$cluster <- factor(kmeans_model$cluster)

# Plot clusters
ggplot(tg2, aes(x = income, y = spend_score, colour = cluster)) +
  geom_point(alpha = 0.7, size = 3) +
  labs(title = "Customer Segmentation using K-Means",
       x = "Loyalty Points",
       y = "Spend Score") +
    theme_minimal()


############################################

