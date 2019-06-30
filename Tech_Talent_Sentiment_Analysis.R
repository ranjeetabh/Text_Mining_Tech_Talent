# This project is a continuation to the earlier text mining project where employee reviews from Amazon and 
# Google were analyzed (https://github.com/ranjeetabh/Text_Mining_Tech_Talent/blob/master/Tech_Talent_Sourcing.R).
# Post text mining, sentiment analysis will be done here. Positive and negative sentiment words for 
# individual dataframes will be analyzed comparing with different standard lexicons.
#################################################################################################################

## Importing necessary libraries
library(qdap)
library(ggplot2)
library(ggthemes)
library(tidytext)
library(tm)
library(tidyr)
library(dplyr)
library(textdata)


# Retrieving the current working directory
getwd()

# Setting current working directory based on file path. This step should be modified for inserting dynamic file paths.
setwd('/Users/ranjeeta/Desktop/Semester 2 Subjects/Data Science and Machine Learning/Text Mining')

# Loading input file. Ensuring reviews are loaded as character vectors and not factors.
amzn_reviews <- read.csv('500_amzn.csv', stringsAsFactors = FALSE)
goog_reviews <- read.csv('500_goog.csv', stringsAsFactors = FALSE)

# Printing Structure and other details of the review files
str(amzn_reviews)
str(goog_reviews)
glimpse(amzn_reviews)
glimpse(goog_reviews)
names(amzn_reviews)
names(goog_reviews)

# Printing Dimensions of the dataset
dim(amzn_reviews)
dim(goog_reviews)

# Calculating quick polarity. Applying polarity() to first 6 reviews of the datasets
practice_pol_amzn_pros <- polarity(amzn_reviews$pros[1:6])
practice_pol_amzn_cons <- polarity(amzn_reviews$cons[1:6])
practice_pol_goog_pros <- polarity(goog_reviews$pros[1:6])
practice_pol_goog_cons <- polarity(goog_reviews$cons[1:6])


# Reviewing the objects
practice_pol_amzn_pros
practice_pol_amzn_cons
practice_pol_goog_pros
practice_pol_goog_cons


# Summarizing the polarity scores
summary(practice_pol_amzn_pros$all$polarity)
summary(practice_pol_amzn_cons$all$polarity)
summary(practice_pol_goog_pros$all$polarity)
summary(practice_pol_goog_cons$all$polarity)


# Plotting polarity all elements
ggplot(practice_pol_amzn_pros$all, aes(x = polarity, y = ..density..)) +
  geom_histogram(binwidth = 0.25, fill = "#bada55", colour = "grey60") +
  geom_density(size = 0.75) +
  theme_gdocs() 

ggplot(practice_pol_amzn_cons$all, aes(x = polarity, y = ..density..)) +
  geom_histogram(binwidth = 0.25, fill = "#bada55", colour = "grey60") +
  geom_density(size = 0.75) +
  theme_gdocs() 

ggplot(practice_pol_goog_pros$all, aes(x = polarity, y = ..density..)) +
  geom_histogram(binwidth = 0.25, fill = "#bada55", colour = "grey60") +
  geom_density(size = 0.75) +
  theme_gdocs() 

ggplot(practice_pol_goog_cons$all, aes(x = polarity, y = ..density..)) +
  geom_histogram(binwidth = 0.25, fill = "#bada55", colour = "grey60") +
  geom_density(size = 0.75) +
  theme_gdocs() 

# Accessing different lexicons/dictionaries
bing <- get_sentiments("bing")
afinn <- get_sentiments("afinn")
loughran <- get_sentiments("loughran")


# Creating a corpus of both positive and negative terms by concatenating for amazon and google

all_corpus_amz <- c(amzn_reviews$pros, amzn_reviews$cons) %>% 
  # Source from a vector
  VectorSource() %>% 
  # Create a volatile corpus
  VCorpus()

all_corpus_goog <- c(goog_reviews$pros, goog_reviews$cons) %>% 
  # Source from a vector
  VectorSource() %>% 
  # Create a volatile corpus
  VCorpus()


## Creating a term-document matrix from all_corpus
# Using term frequency inverse document frequency weighting by setting weighting to weightTfIdf
# Instead of counting the number of times a word is used (frequency), the values in the TDM are penalized for over used terms, 
# which helps reduce non-informative words.

all_tdm_amzn <- TermDocumentMatrix(
  # Use all_corpus
  all_corpus_amz, 
  control = list(
    # Use TFIDF weighting
    weighting = weightTfIdf, 
    # Remove the punctuation
    removePunctuation = TRUE,
    # Use English stopwords
    stopwords = stopwords(kind = "en")
  )
)


all_tdm_goog <- TermDocumentMatrix(
  # Use all_corpus
  all_corpus_goog, 
  control = list(
    # Use TFIDF weighting
    weighting = weightTfIdf, 
    # Remove the punctuation
    removePunctuation = TRUE,
    # Use English stopwords
    stopwords = stopwords(kind = "en")
  )
)

# Analysing the TDM's created for Amazon and Google
all_tdm_amzn
all_tdm_goog

## Creating a tidy text tibble for Amazon 
# Vector to tibble
tidy_reviews_amzn_pros <- amzn_reviews %>% 
  unnest_tokens(word, pros)

tidy_reviews_amzn_cons <- amzn_reviews %>% 
  unnest_tokens(word, cons)

# Creating a merged tidy text tibble for pros and cons for Amazon
all_reviews_amz <- bind_rows(tidy_reviews_amzn_pros, tidy_reviews_amzn_cons)

head(all_reviews_amz)

# Analyzing with lexicons and finding top positive and negative words for Amazon and Google
word_counts <- all_reviews_amz %>%
  # Implement sentiment analysis using the "bing"/loughran/afinn lexicon
  inner_join(get_sentiments("bing")) %>%
  #inner_join(get_sentiments("loughran")) %>%
  #inner_join(get_sentiments("afinn")) %>%
  # Count by word and sentiment
  count(word, sentiment, sort = TRUE)

top_words <- word_counts %>%
  # Group by sentiment
  group_by(sentiment) %>%
  # Take the top 10 for each sentiment
  top_n(10) %>%
  ungroup() %>%
  # Make word a factor in order of n
  mutate(word = reorder(word, n))

# Use aes() to put words on the x-axis and n on the y-axis
ggplot(top_words, aes(word, n, fill = sentiment)) +
  # Make a bar chart with geom_col()
  geom_col(show.legend = FALSE) +
  facet_wrap(~sentiment, scales = "free") +  
  coord_flip() + ggtitle("Amazon Word Sentiments")

amazon_overall_sentiment <- all_reviews_amz %>%
  inner_join(get_sentiments("bing")) %>% # pull out only sentiment words
  count(sentiment) %>% # count the # of positive & negative words
  spread(sentiment, n, fill = 0) %>% # made data wide rather than narrow
  mutate(sentiment = positive - negative) # # of positive words - # of negati
  
head(amazon_overall_sentiment)

print("Overall sentiment of Amazon is positive")

## Creating a tidy text tibble for Google 
# Vector to tibble
tidy_reviews_goog_pros <- goog_reviews %>% 
  unnest_tokens(word, pros)

tidy_reviews_goog_cons <- goog_reviews %>% 
  unnest_tokens(word, cons)


# Creating a merged tidy text tibble for pros and cons for Google
all_reviews_goog <- bind_rows(tidy_reviews_goog_pros, tidy_reviews_goog_cons)

head(all_reviews_goog)

word_counts <- all_reviews_goog %>%
  # Implement sentiment analysis using the "bing" lexicon
  inner_join(get_sentiments("bing")) %>%
  #inner_join(get_sentiments("loughran")) %>%
  #inner_join(get_sentiments("afinn")) %>%
  # Count by word and sentiment
  count(word, sentiment, sort = TRUE)

top_words <- word_counts %>%
  # Group by sentiment
  group_by(sentiment) %>%
  # Take the top 10 for each sentiment
  top_n(10) %>%
  ungroup() %>%
  # Make word a factor in order of n
  mutate(word = reorder(word, n))

# Use aes() to put words on the x-axis and n on the y-axis
ggplot(top_words, aes(word, n, fill = sentiment)) +
  # Make a bar chart with geom_col()
  geom_col(show.legend = FALSE) +
  facet_wrap(~sentiment, scales = "free") +  
  coord_flip() + ggtitle("Google Word Sentiments")


google_overall_sentiment <- all_reviews_goog %>%
  inner_join(get_sentiments("bing")) %>% # pull out only sentiment words
  count(sentiment) %>% # count the # of positive & negative words
  spread(sentiment, n, fill = 0) %>% # made data wide rather than narrow
  # # of positive words - # of negati
  mutate(sentiment = positive - negative) 
  
head(google_overall_sentiment)

print("Overall sentiment of Google is positive")

print("Boxplot of Amazon (1) and Google sentiment score (2)")
boxplot(amazon_overall_sentiment$sentiment,google_overall_sentiment$sentiment)
print("Overall sentiment of Google is more positive than Amazon")




