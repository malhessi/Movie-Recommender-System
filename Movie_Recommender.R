##########################################################
# Create edx set, validation set (final hold-out test set)
##########################################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(data.table)

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
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           title = as.character(title),
                                           genres = as.character(genres))
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

#================================================================

# No. rows and columns in edx dataset
nrow(edx)
nrow(validation)
ncol(edx)

n_zeros <- edx %>% filter(rating==0) %>% summarize(n_zeros=n())
##
table(edx$rating)

## Distinct Movies (10677)
# use base function
length(unique(edx$movieId))

# use tidyverse (dplyr)
edx %>% distinct(movieId) %>% summarize(n_dist = n())
edx %>% select(movieId) %>% n_distinct()

## Distinct Users (69878)
edx %>% select(userId) %>% n_distinct()

## genres
table(edx$genres)

edx %>% filter(genres=="Drama") %>% summarize(n())
edx %>% filter(genres=="Comedy") %>% summarize(n())
edx %>% filter(genres=="Thriller") %>% summarize(n())
edx %>% filter(genres=="Romance") %>% summarize(n())

edx %>% filter(str_detect(genres,"Drama")) %>% summarize(n())
edx %>% filter(str_detect(genres,"Comedy")) %>% summarize(n())
edx %>% filter(str_detect(genres,"Thriller")) %>% summarize(n())
edx %>% filter(str_detect(genres,"Romance")) %>% summarize(n())

edx %>% group_by(movieId, title) %>% summarize(n=n()) %>% arrange(desc(n)) %>% ungroup()

table(edx$rating)

# Plot the histogram of the ratings of all users and all movies
edx %>% ggplot(aes(x=rating)) + geom_histogram() + geom_vline(xintercept =mean(edx$rating), color="red" ) + geom_vline(xintercept =median(edx$rating), color="blue")

# plot the histogram of the ratings of all users for each movie
edx %>% ggplot(aes(x=movieId)) + geom_histogram() + scale_x_log10()

# Visualize the variability of the movies and users
edx %>% 
  dplyr::count(movieId) %>%
  ggplot(aes(n)) + 
  geom_histogram(bins = 30, color = "black") + 
  scale_x_log10() + 
  ggtitle("Movies")

edx %>%
  dplyr::count(userId) %>% 
  ggplot(aes(n)) + 
  geom_histogram(bins = 30, color = "black") + 
  scale_x_log10() +
  ggtitle("Users")

# Display the top ten rated movies
edx %>% count(movieId, title, userId) %>% top_n(10,n) %>% arrange(desc(n))

# RMSE Function
RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

# We have two datasets:
# EDX
# Validation

# Partition EDX Dataset into Training and Test Datasets
set.seed(1985)
test_indices <- createDataPartition(edx$rating, times = 1, p=0.25, list=FALSE)
train_edx <- edx[-test_indices,]
test_edx <- edx[test_indices,]

#  Remove the users and movies that don't appear in both test and training datasets.
test_edx <- test_edx %>% semi_join(train_edx,by="movieId") %>%
  semi_join(train_edx, by="userId")

#==================================================================
# 10 points: 0.86550 <= RMSE <= 0.89999
# 15 points: 0.86500 <= RMSE <= 0.86549
# 20 points: 0.86490 <= RMSE <= 0.86499
# 25 points: RMSE < 0.86490

#================== First Model = Constant Mean ===================

# Y(u,i) = Mu + E(u,i) u: user, i: movie
mu_hat <- mean(edx$rating) # mu_hat=3.5125

# Calculate RMSE
rmse_muhat <- RMSE(validation$rating, mu_hat) #RMSE=1.06120

#================== Second Model = Movie Effect ===================
# Y(u,i) = Mu + b(i) + E(u,i) u:user, i:movie, b:movie effect
# b(i) = Y(u,i) - Mu
movie_effect <- edx %>% group_by(movieId) %>% summarize(bi = mean(rating-mu_hat))

predicted_ratings_m <- validation %>% left_join(movie_effect, by="movieId") %>% mutate(pred_m = mu_hat + bi)
# Calculate RMSE
rmse_bi <- RMSE(validation$rating, predicted_ratings_m$pred_m) #RMSE=0.94391

#================== Third Model = User Effect ===================
# Y(u,i) = Mu + b(i) + b(u) + E(u,i) u:user, i:movie, b:movie effect
# b(u) = Y(u,i) - Mu - b(i)
user_effect <- edx %>% left_join(movie_effect, by="movieId") %>% group_by(userId) %>% summarize(bu = mean(rating-mu_hat-bi))

predicted_ratings_u <- validation %>% left_join(movie_effect, by="movieId") %>% left_join(user_effect, by="userId") %>% mutate(pred_u = mu_hat + bi + bu)
# Calculate RMSE
rmse_bu <- RMSE(validation$rating, predicted_ratings_u$pred_u) #RMSE=0.86535

#=========Fourth Model =  Matrix Factorization with Regularization =========




train_small <- edx %>% 
  group_by(movieId) %>%
  filter(n() >= 50 | movieId == 3252) %>% ungroup() %>% 
  group_by(userId) %>%
  filter(n() >= 50) %>% ungroup()

y <- train_small %>% 
  select(userId, movieId, rating) %>%
  pivot_wider(names_from = "movieId", values_from = "rating") %>%
  as.matrix()

rownames(y)<- y[,1]
y <- y[,-1]

movie_titles <- edx %>% 
  select(movieId, title) %>%
  distinct()

colnames(y) <- with(movie_titles, title[match(colnames(y), movieId)])


y <- sweep(y, 2, colMeans(y, na.rm=TRUE))
y <- sweep(y, 1, rowMeans(y, na.rm=TRUE))

m_1 <- "Godfather, The (1972)"
m_2 <- "Godfather: Part II, The (1974)"
p1 <- qplot(y[ ,m_1], y[,m_2], xlab = m_1, ylab = m_2)

m_1 <- "Godfather, The (1972)"
m_3 <- "Goodfellas (1990)"
p2 <- qplot(y[ ,m_1], y[,m_3], xlab = m_1, ylab = m_3)

m_4 <- "You've Got Mail (1998)" 
m_5 <- "Sleepless in Seattle (1993)" 
p3 <- qplot(y[ ,m_4], y[,m_5], xlab = m_4, ylab = m_5)

gridExtra::grid.arrange(p1, p2 ,p3, ncol = 3)