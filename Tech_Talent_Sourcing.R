# This project is a case study on HR Analytics for employees working in Google and Amazon. Online 
# employee reviews have been collected from "https://www.glassdoor.com/index.htm". The reviews
# are anonymous to protect privacy. From this corpus, we will try to answer questions pertaining to 
# better work-life balance, better perceived pay and other interesting insights.
# Input file contains reviews obtained from Glassdoor and are tagged under headers pros and cons.
# This Text Mining project is based on  "Bag of words" model where all the words are analysed
# as a single token and order doesn't matter. The cleaned CSV files are collected from Datacamp platform.
# Employee review text data is distributed under header "pros" and "cons" in the input csv files.

#######################################################################################################
# Loading the readr and dplyr packages
library(readr)
library(dplyr)

# Retrieving the current working directory
getwd()

# Setting current working directory based on file path. This step should be modified for inserting dynamic file paths.
setwd('/Users/ranjeeta/Desktop/Semester 2 Subjects/Data Science and Machine Learning/Text Mining')

# Importing the organization data
amzn_revs <- read_csv("500_amzn.csv")
goog_revs <- read_csv("500_goog.csv")

# Printing the structure of amazon reviews
str(amzn_revs)
glimpse(amzn_revs)
names(amzn_revs)

# Printing the structure of google reviews
str(goog_revs)
glimpse(goog_revs)
names(goog_revs)

# Creating postive and negative reviews dataframe objects individually for each companies

# For Amazon
amzn_pros <- amzn_revs$pros
amzn_cons <- amzn_revs$cons

# For Google
goog_pros <- goog_revs$pros
goog_cons <- goog_revs$cons

# Text Organization
####################

# Implementing necessary libraries 
library(qdap)
library(tm)

# Defining custom text cleaning function qdap_clean and tm_clean
# qdap functions can be applied directly to a text vector than a corpus object. x is the input vector.
# tm functions can be applied on a corpus

qdap_clean <- 
  function(x) {
  x <- replace_abbreviation(x)
  x <- replace_contraction(x)
  x <- replace_number(x)
  x <- replace_ordinal(x)
  x <- replace_symbol(x)
  x <- tolower(x)
  return(x)
}

tm_clean <- function(corpus) {
  tm_clean <- tm_map(corpus, removePunctuation)
  corpus <- tm_map(corpus, stripWhitespace)
  corpus <- tm_map(corpus, removeWords,
                   # Removal of relevant stopwords and context specific words
                   c(stopwords("en"), "Google","google","amazon","Amazon","company","can","i","im"))
  return(corpus)
}

# Invoking qdap_clean function to clean positive and negative review dataframes for Amazon and Google
qdap_cleaned_amzn_pros <- qdap_clean(amzn_pros)
qdap_cleaned_amzn_cons <- qdap_clean(amzn_cons)
qdap_cleaned_goog_pros <- qdap_clean(goog_pros)
qdap_cleaned_goog_cons <- qdap_clean(goog_cons)


# Creating the corpus
# Corpus is a collection of text documents
# Vectorsource is used as source object here which inputs only character vectors
amzn_p_corp <- VCorpus(VectorSource(qdap_cleaned_amzn_pros))
amzn_c_corp <- VCorpus(VectorSource(qdap_cleaned_amzn_cons))
goog_p_corp <- VCorpus(VectorSource(qdap_cleaned_goog_pros))
goog_c_corp <- VCorpus(VectorSource(qdap_cleaned_goog_cons))


# Invoking tm_clean function to clean the corpus created in previous step
amzn_pros_corp <- tm_clean(amzn_p_corp)
amzn_cons_corp <- tm_clean(amzn_c_corp)
goog_pros_corp <- tm_clean(goog_p_corp)
goog_cons_corp <- tm_clean(goog_c_corp)


# Stemming all corpus documents
# In linguistics, stemming is the process of reducing inflected (or derived) words to their word stem,
# base or root form-generally a written word form.

# Importing necessary library
library(SnowballC)

amzn_pros_corp = tm_map(amzn_pros_corp, stemDocument)
amzn_cons_corp = tm_map(amzn_cons_corp, stemDocument)
goog_pros_corp = tm_map(goog_pros_corp, stemDocument)
goog_cons_corp = tm_map(goog_cons_corp, stemDocument)

# Quick viewing of the corpus content
# The corpus object in R is a nested list.
amzn_pros_corp[[8]][1]
goog_cons_corp[[10]][1]

# Finding the 20 most frequent terms in the corpus amzn_pros_corp
term_count <- freq_terms(amzn_pros_corp, 20)

# Plotting 20 most frequent terms for amzn_pros_corp
plot(term_count)

# Feature Extraction and Analysis
#################################

# Importing necessary libraries
library(RWeka)

# Creating custom function tokenizer
token_delim <- " \\t\\r\\n.!?,;\"()"

tokenizer <- function(x) {
  NGramTokenizer(x, Weka_control(min = 2, max = 2, delimiters = token_delim))}

# A document-term matrix is a mathematical matrix that describes the frequency of terms that occur 
# in a collection of documents. In a document-term matrix, rows correspond to documents in the 
# collection and columns correspond to terms. The term-document matrix is a transpose of the 
# document-term matrix. It is generally used for language analysis. An easy way to start analyzing 
# the information is to change the DTM/TDM into a simple matrix using as.matrix().

# Creating Term Document Matrix (TDM) for Amazon pros reviews

amzn_p_tdm <- TermDocumentMatrix(
  amzn_pros_corp, 
  control = list(tokenize = tokenizer)
)

# Converting TDM to Matrix form
amzn_p_tdm_m <- as.matrix(amzn_p_tdm)

# Computing term frequencies
amzn_p_freq <- rowSums(amzn_p_tdm_m)

# Sorting term frequency in decreasing order
term_frequency <- sort(amzn_p_freq, decreasing = TRUE)

# Viewing the top 5 most frequently occuring bigrams
# Here bigram (combination of two words is created by passing appropriate argument in NGramTokenizer)
term_frequency[1:5]

# Plotting a barchart of the 5 most common bigrams
barplot(term_frequency[1:5], col = "steel blue", las = 2)


# Plotting a wordcloud using amzn_p_freq values
# Importing necessary library
suppressWarnings(library(wordcloud))

wordcloud(names(amzn_p_freq), amzn_p_freq, 
          max.words = 20, color = "blue")

# Printing the word cloud with the specified colors to distinguish between more and less frequent words
wordcloud(names(amzn_p_freq), amzn_p_freq,
          max.words = 20, colors = c("aquamarine","darkgoldenrod","tomato"))

# Creating Term Document Matrix (TDM) for Amazon cons reviews
amzn_c_tdm <- TermDocumentMatrix(
  amzn_cons_corp, 
  control = list(tokenize = tokenizer)
)

# Converting TDM to Matrix form
amzn_c_tdm_m <- as.matrix(amzn_c_tdm)

# Computing term frequencies
amzn_c_freq <- rowSums(amzn_c_tdm_m)

# Sorting in decreasing order of frequency
term_frequency <- sort(amzn_c_freq, decreasing = TRUE)

# Viewing the top 5 most frequntly occuring bigrams
term_frequency[1:5]

# Plotting a wordcloud using amzn_c_freq values
wordcloud(names(amzn_c_freq), amzn_c_freq, 
          max.words = 25, color = "red")

# Plotting Dendrogram for Amazon cons reviews
# Hierarchical clustering technique is used to check how connected the top relevant words are

# Creating amzn_c_tdm
amzn_c_tdm <- TermDocumentMatrix(
  amzn_cons_corp,
  control = list(tokenize = tokenizer)
)

# Printing amzn_c_tdm to the console
amzn_c_tdm

# Creating amzn_c_tdm2 by removing sparse terms 
amzn_c_tdm2 <- removeSparseTerms(amzn_c_tdm, .993)

# Creating hc as a cluster of distance values
hc <- hclust(dist(amzn_c_tdm2), 
             method = "complete")

# Producing a plot of hc
# hc plot shows how certain words are grouped together, set of words used together most frequently
plot(hc)


# Word association is a way of calculating the correlation between 2 words in a DTM or TDM. It is another
# way of identifying words used together frequently. Here we are finding association with "fast paced", 
# "good pay" etc. for amazon pros tdm
associations <- findAssocs(amzn_p_tdm, "fast paced", 0.2)

# Creating associations_df
associations_df <- list_vect2df(associations)[, 2:3]

# Importing plotting library
library(ggplot2)
library(ggthemes)

# Plotting the associations_df values 
ggplot(associations_df, aes(y = associations_df[, 1])) + 
  geom_point(aes(x = associations_df[, 2]), 
             data = associations_df, size = 3) + 
  ggtitle("Word Associations to 'fast paced'") + 
  theme_gdocs()


## Comparison of Amazon with Google
###################################

# Combining both pros and cons corpora of Google reviews and forming a single object
all_yes_goog <- paste(qdap_cleaned_goog_pros, collapse = "")
all_no_goog <- paste(qdap_cleaned_goog_cons, collapse = "")
all_goog_comb <- c(all_yes_goog, all_no_goog)
all_goog_corp <- VCorpus(VectorSource(all_goog_comb))
all_goog_corp <- tm_clean(all_goog_corp)
all_goog_corp <- tm_map(all_goog_corp, stemDocument)


# Creating TDM
all_goog_tdm <- TermDocumentMatrix(all_goog_corp)

# Creating matrix
all_goog_m <- as.matrix(all_goog_tdm)

# Building a comparison cloud to identify dissimilar words used between google pros and cons corpora
comparison.cloud(all_goog_m, 
                 #colors = c("#F44336", "#2196f3"), 
                 colors = c("green", "red"), 
                 max.words = 100)

# Building a commonality cloud to identify similar words used between google pros and cons corpora
commonality.cloud(all_goog_m, 
                 colors = "steelblue1", 
                 max.words = 100)


## Comparing Amazon with Google Pro reviews
# Creating a Pyramid plot to compare and check top bigrams shared between two companies

# Combining both pros and cons corpora of Amazon reviews and forming a single object
all_yes <- paste(qdap_cleaned_amzn_pros, collapse = "")
all_no <- paste(qdap_cleaned_amzn_cons, collapse = "")
all_amzn_comb <- c(all_yes, all_no)
all_amzn_corp <- VCorpus(VectorSource(all_amzn_comb))
all_amzn_corp <- tm_clean(all_amzn_corp)
all_amzn_corp <- tm_map(all_amzn_corp, stemDocument)

# Creating TDM
all_amzn_tdm <- TermDocumentMatrix(all_amzn_corp)

# Creating matrix
all_amzn_m <- as.matrix(all_amzn_tdm)

# Creating final combined dataframe of amazon and google pros and cons reviews
# Dataframe consisting of terms and corresponding AmazonPro, and GooglePro word frequencies.
all_tdm_m <- rbind(all_amzn_m, all_goog_m)

# Importing necessary library
library(plotrix)

# Identifying words shared by both documents
common_words <- subset(all_tdm_m, all_tdm_m[, 1] > 0 & all_tdm_m[, 2] > 0)

# Calculating common words and difference
difference <- abs(common_words[, 1] - common_words[, 2])
common_words <- cbind(common_words, difference)
common_words <- common_words[order(common_words[, 3],
                                   decreasing = TRUE), ]

head(common_words)

# Creating dataframe of top 15 words
top15_df <- data.frame(x = common_words[1:15, 1],
                       y = common_words[1:15, 2],
                       labels = rownames(common_words[1:15, ]))

# Making Pyramid plot
pyramid.plot(top15_df$x, top15_df$y, labels = top15_df$labels,
             gap = 40, main = "Words in Common", unit = NULL,
             top.labels = c("Amzn", "Common Words", "Google"))


