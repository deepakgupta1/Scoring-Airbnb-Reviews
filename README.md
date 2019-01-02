# Scoring-Airbnb-Reviews
Score prediction in the categories of cleanliness, accuracy, check-in, communication, location, value (out of 10) and overall (out of 100) based on the reviews left by customers on Airbnb. Find more about the challenge [here](https://community.topcoder.com/longcontest/?module=ViewProblemStatement&rd=17202&pm=14964).

This is my award winning position writeup for the challenge.

### 1. Problem and data.
Doing sentiment analysis on reviews of comments left by customers and predict seven different score based on the reviews and other data given.
Lots of data was given:
* Data about reviews of customers.
* Data about the listing properties.
* Data about neighborhood of listings.
* Data about pricing of listings.

### 2. Data Preprocessing.
Since major part of data was text, some preprocessing was required. Some of the major I did:
* Cleaning reviews text: Extracting only alphabets and numbers, removing stopwords and so on.
* Filling null values: Some columns in listings data had missing values like in host_response_rate, host_response_time, bedrooms, bathrooms and so on. Null values in train labels (rating, value, cleanliness etc.) were filled with mean of the values and then rounded off to integers.
* Cleaning some columns: Some columns like price, security_deposit, extra_people had special character like %, $, these were removed, and values were changed to float.

### 3. Feature Engineering.
I did considerable feature engineering, mostly on the text features. Most of these are:
* Sentiment: For reviews, description of listing. Each sentiment score has four different values: positive, negative, neutral and compound (overall sentiment).
* Tfidf features: Made tfidf vectors from reviews of 500 length.
* Countvectorizer features: Made countvectorizer features from reviews of 500 length.
* Word2vec features:
  * Self-trained a word2vec model on the cleaned reviews.
  * Made word2vec features of length 500 for each review by taking mean of vectors of all words in the cleaned reviews.
* Target encoded features: Since there were some categorical features like property_type, room_type, cancellation_policy, I target encoded these features. Made mean target values by each different value of the categorical columns and then merged with both train and test data.
* Features on listings:
  * Days since first and last review for each listing.
  * Reviews per month for each listing.
  * Sentiment scores for name, description, neighbourhood_overview, notes, space, summary for each listing.
* Features on reviews:
  * Length of review.
  * Number of words in review.

### 4. Algorithm and modelling.
Taking the amount of data in to consideration, I choose LightGBM Regressor to be modelling algorithm. This algorithm works very fast and achieves really good performance too.

### 5. Hyperparameters tuning and cross-validation strategy.
#### Cross-validation strategy: 
10-fold stratified cross-validation, stratified over the label (rating, value, cleanliness etc.). Since the ratings were mostly approaching their maximum values (100, 10), so the distribution of the ratings was very skewed that’s why `stratified over the ratings` itself. This way there was a constancy in the cross-validation scores among the 10 folds.
#### Evaluation metric: 
RMSE. The overall RMSE score for a 10-fold cross-validation was the mean of all 10 scores.
#### Hyperparameter tuning: 
Used `Bayesian Optimization` algorithm for parameters tuning of the lightgbm algorithm using the above described cross-validation strategy.

### 6. Training (using IBM Watson)
* Made a project on IBM Watson.
* Uploaded the processed and feature added train and test dataset as assets to the project.
* Started a jupyter notebook in the project, loaded the datasets.
* Wrote python code for training the lightgbm model with the optimized parameters.
* Installed lightgbm in the jupyter environment for use.
* Training was done on whole training dataset.

### 7. Code files
* Tfidf_countved_features: Cleans the reviews text and makes tfidf and countvec features and some other features.
* Train_word2vec: Trains a word2vec model on the cleaned reviews and saves the model.
* Make_word2ved_features: Makes word2vec features for reviews from the trained word2vec model.
* Listing_features: Makes features from listing data.
* Lgbm_training: Trains lgbm model with the tuned parameters on the entire training data.

### 8. Technical Specifications:
* Python 2.7
* Lightgbm.
* Sklearn.
* Bs4
* Genism
* Nltk
* Pandas
* Numpy
