##### Authors: Jeshuran Thangaraj, Vaishnavi Mukundhan, Prateek Srivastava

### Sentiment Analysis:
Sentiment analysis or Opinion Mining is a process of extracting the opinions in a text rather than the topic of the document.  The text would have sentences that are either facts or opinions. We classify the opinions into three categories: Positive, Negative and Neutral.

### Datasets:
The existing implementation<sup>[1]</sup>  uses the Kaggle Dataset. In our implementation we have used the below two datasets:

1. Kaggle IMDB
2. Large Movie Review Dataset (Stanford Movie Review Dataset)

### Existing Implementations:
 + SentiWordNet<sup>[2]</sup>:
  * SentiWordNet is a resource used to perform Opinion Mining.
  * This resource considers the synset instead of the term as each of the sense would have different opinions.
  * It gives a positive, negative and objective score to each word which totals to 1.
  * SentiWordNet uses Eight ternary classifiers to find the opinion of each word.

+  Naives Bayes to Classify the Opinions<sup>[3]</sup>:
  * In this approach, they extract a feature vector from the text and directly classify them using  the NLTK's Naive Bayes Classifier.
  * This can also be done with other  classifiers like SVM.

### Our Approaches:
After reviewing multiple approaches, we finally narrowed down on the following two implementations which could be reviewed in the time available.

* BiGram Model:
  - This is an add on to the pre existing unigram model implementation<sup>[1]</sup>. In this we first pre process the text by cleansing, Tokenizing, Lemmatizing and POS tagging.
  - Each word is given a score between 0-1 by passing it to the SENTIWORDNET.
  - We changed the scoring mechanism from a unigram model to a bigram model by taking the MAX absolute score of either model.
  - Accuracy :
    - Bigram 61%
    - Unigram 66%  


* Building a WordNet for our Datasets:
  - We started under the assumption that training on similar data will yield better results.
  - Trained a unigram model using 25000 movie reviews.
  - We extracted the ADJ and ADV POS-tags from the training corpus and built a frequency distribution for each word based on its occurrence in positive and negative reviews.
  - It was then used on our test set to predict opinions.
  - Accuracy:
    - Negative Test set 75.4%  
    - Positive Test set 67%

### Future Approaches:
  - We would like to figure out why the Bigram model is giving us a lesser accuracy than the unigram model.
  - For the second approach we would use Bigram
  - We would also like to try using LSTM.  

### Code Link
 + https://github.com/prateek22sri/sentimentAnalysis

### References:
+ http://nlpforhackers.io/sentiment-analysis-intro/
+ http://sentiwordnet.isti.cnr.it/
+ https://streamhacker.com/tag/bigrams/
 
