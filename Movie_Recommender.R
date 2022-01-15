###################################################################
#  title: "Movies Recommender System - based on Movielens dataset"#
#  author: "Mohammed Alhessi"                                     #
#  date: "12/29/2021"                                             #
###################################################################

# Loading and installing packages
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(recosystem)) install.packages("recosystem", repos = "http://cran.us.r-project.org")
if(!require(recommenderlab)) install.packages("recommenderlab", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(data.table)
library(recosystem)
library(recommenderlab)

# Set the target RMSE
target <- 0.86490

# Set Significant Digits Option
options(pillar.sigfigs=6)

##########################################################
# Create edx set, validation set (final hold-out test set)
##########################################################

# Note: this process could take a couple of minutes

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

# if using R 3.6 or earlier:
# movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
#                                            title = as.character(title),
#                                            genres = as.character(genres))
# if using R 4.0 or later:
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))


movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

##########################################################
## Explore and Clean the data
##########################################################

# Print sample of 10 ratings
knitr::kable(sample_n(edx,10,replace=TRUE ))

# Create summary of the dataset edx
summary(edx)

# Distribution of ratings
edx %>% ggplot(aes(x=rating)) +
  geom_bar(fill="#006bbd") +
  geom_text(aes(label=..count..), stat="count", color="white", vjust=1.5) +
  geom_vline(xintercept =mean(edx$rating), color="red" ) +
  geom_vline(xintercept =median(edx$rating), color="blue")


# Number of users
nusers <- edx %>% select(userId) %>% n_distinct()

# Number of movies
nmovies <- edx %>% select(movieId) %>% n_distinct()


# Take a sample 100 distinct users with unique ids
sample_users <- sample(unique(edx$userId), 100)

# Create and plot a matrix of 100 different users who rated a sample of
# 100 movies. This is to visualize part of the large sparse matrix.
x<-edx %>% filter(userId %in% sample_users) %>%
  select(userId, movieId, rating) %>%
  spread(movieId, rating) %>%
  select(sample(ncol(.), 100)) %>% 
  as.matrix() %>% t(.) %>%
  image(1:100, 1:100,. , xlab="Movies", ylab="Users")


### Explore the distribution/ variability of the users

edx %>%
  group_by(userId) %>%
  summarize(n=n()) %>%
  ggplot(aes(n)) + 
  geom_histogram(bins = 30, color = "black") + 
  scale_x_log10() +
  ggtitle("Users")

### Explore the distribution of the movies

edx %>%
  group_by(movieId) %>%
  summarize(n=n()) %>%
  ggplot(aes(n)) + 
  geom_histogram(bins = 30, color = "black") + 
  scale_x_log10() +
  ggtitle("Movies")

##########################################################
## Building the Model
##########################################################


# RMSE Function
RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}


### Partition the dataset `edx`
# Partition edx Dataset into Training and Test Datasets
set.seed(1985)
test_indices <- createDataPartition(edx$rating, times = 1, p=0.25, list=FALSE)
train_edx <- edx[-test_indices,]
test_edx <- edx[test_indices,]

#  Remove the users and movies that don't appear in both test and training datasets.
test_edx <- test_edx %>% semi_join(train_edx,by="movieId") %>%
  semi_join(train_edx, by="userId")


### Average Model
#============= First Model = Constant Mean =============
# Y(u,i) = Mu + E(u,i) u: user, i: movie
mu_hat <- mean(edx$rating) # mu_hat=3.5125

# Calculate RMSE
rmse_muhat <- RMSE(validation$rating, mu_hat) #RMSE=1.06120

# Store the result
accuracy_results <- tibble(Model = "Average_Model", RMSE = rmse_muhat)


### Movie Effect Model
#-----------------------------------
# ============= Second Model = Movie Effect  =============
# Y(u,i) = Mu + b(i) + E(u,i) u:user, i:movie, b:movie effect
# b(i) = Y(u,i) - Mu
movie_effect <- edx %>% group_by(movieId) %>% summarize(bi = mean(rating-mu_hat))

# Calculate the predicted ratings after considering the movie effect.
predicted_ratings_mv <- validation %>%
  left_join(movie_effect, by="movieId") %>%
  mutate(pred_mv = mu_hat + bi)

# Calculate RMSE
rmse_bi <- RMSE(validation$rating, predicted_ratings_mv$pred_mv)

# Store the result
accuracy_results <- bind_rows(accuracy_results,
                              tibble(Model = "Movie_Effect_Model", RMSE = rmse_bi))

### Movie and User Effects Model
#-----------------------------------
# =============  Third Model = User Effect ============= 
# Y(u,i) = Mu + b(i) + b(u) + E(u,i) u:user, i:movie, b:movie effect
# b(u) = Y(u,i) - Mu - b(i)
user_effect <- edx %>%
  left_join(movie_effect, by="movieId") %>%
  group_by(userId) %>%
  summarize(bu = mean(rating-mu_hat-bi))

predicted_ratings_ur <- validation %>%
  left_join(movie_effect, by="movieId") %>%
  left_join(user_effect, by="userId") %>%
  mutate(pred_ur = mu_hat + bi + bu)

# Calculate RMSE
rmse_bu <- RMSE(validation$rating, predicted_ratings_ur$pred_ur)

# Store the result
accuracy_results <- bind_rows(accuracy_results,
                              tibble(Model = "User_Effect_Model", RMSE = rmse_bu))

### Regularized Movie and User Effects Model
#-----------------------------------
# =====  Fourth Model = Regularized Movie and User Effects =====

# Lambda Parameter Tuning
lambdas <- seq(0, 10, 0.25)

rmses <- sapply(lambdas, function(lambda){
  
  mu_hat <- mean(train_edx$rating)
  
  b_i_hat <- train_edx %>% 
    group_by(movieId) %>%
    summarize(b_i_hat = sum(rating - mu_hat)/(n()+lambda))
  
  b_u_hat <- train_edx %>% 
    left_join(b_i_hat, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u_hat = sum(rating - b_i_hat - mu_hat)/(n()+lambda))
  
  predicted_ratings <- 
    test_edx %>% 
    left_join(b_i_hat, by = "movieId") %>%
    left_join(b_u_hat, by = "userId") %>%
    mutate(pred_rate = mu_hat + b_i_hat + b_u_hat) %>%
    pull(pred_rate)
  
  return(RMSE(predicted_ratings, test_edx$rating))
})

qplot(lambdas, rmses)  

# Selected Lambda
sel_lambda <- lambdas[which.min(rmses)]
sel_lambda

# Predict the ratings using the selected Lambda
mu_hat <- mean(edx$rating)

b_i_hat <- edx %>% 
  group_by(movieId) %>%
  summarize(b_i_hat = sum(rating - mu_hat)/(n()+sel_lambda))

b_u_hat <- edx %>% 
  left_join(b_i_hat, by="movieId") %>%
  group_by(userId) %>%
  summarize(b_u_hat = sum(rating - b_i_hat - mu_hat)/(n()+sel_lambda))

predicted_ratings <- 
  validation %>% 
  left_join(b_i_hat, by = "movieId") %>%
  left_join(b_u_hat, by = "userId") %>%
  mutate(pred_rate = mu_hat + b_i_hat + b_u_hat)

# Calculate RMSE
rmse_reg <- RMSE(validation$rating, predicted_ratings$pred_rate)

# Store the result
accuracy_results <- bind_rows(
  accuracy_results,
  tibble(Model = "Regularized Movie_and_User_Effect_Model", RMSE = rmse_reg)
)

    
### Regularized Model with Matrix Factorization
#----------------------------------------------

#### MAtrix Factorization using `recosystem` package

set.seed(19850, sample.kind = "Rounding")

edx_reco <-  with(edx, data_memory(user_index = userId, 
                                   item_index = movieId, 
                                   rating = rating))

validation_reco  <- with(validation, data_memory(user_index = userId, 
                                                 item_index = movieId, 
                                                 rating = rating))

# Create the model object
reco_model <-  recosystem::Reco()

# Tune the parameters
opts <-  reco_model$tune(edx_reco, opts = list(dim = c(10, 20,30), 
                                               lrate = c(0.1, 0.2),
                                               costp_l2 = c(0.01, 0.1), 
                                               costq_l2 = c(0.01, 0.1),
                                               nthread  = 4, niter = 10))

# Train the model
reco_model$train(edx_reco, opts = c(opts$min, nthread = 4, niter = 20))

# Calculate the prediction
predicted_ratings_mf <-  reco_model$predict(validation_reco, out_memory())

# Calculate RMSE
rmse_mf <- RMSE(validation$rating, predicted_ratings_mf)

# Store the result
accuracy_results <- bind_rows(
  accuracy_results,
  tibble(Model = "Matrix Factorization Model", RMSE = rmse_mf)
)

# Print the results of all models
knitr::kable(accuracy_results,
             caption="The implemented Models with thier accuracies.")
