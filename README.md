# Medical_Specialist_Machine_Learning


# Topic Selection


Machine learning is an area that is seeing more acceptance and use in the healthcare industry.  There are many technological advances in handling and analyzing medical data, as it is very clear that it can provide meaningful advances to improve healthcare and therefore the well being of people.  In specific, machine learning algorithms in Natural Language Processing (NLP) are being widely used in Healthcare & Life Sciences to extract information from unstructured text data.  As a group, we decided that we wanted to dip our feet into this fast growing field and apply our machine learning skills to see what we could accomplish!


# Project overview

For our final project, our group chose to use a dataset (from Kaggle) that contained medical transcriptions and the respective medical specialty for around 5000 patients. We chose to implement a supervised machine learning model for Natural Language Processing (NLP) that would predict the medical specialty to which each patient needs based on the medical transcriptions.


# Tools / Technology Used

* CSV File                              
* Pandas                                
* Numpy                                 
* Matplotlib                            
* NLTK (Natural Language Toolkit)
* RandomForest
* Naive Bayes
* Scikit-learn
* LogisticRegression

# Process

Our first (and longest/hardest) task involved cleaning and preprocessing the data we were using.  Our process was a constant give and take as our algorithms success was very dependent on our how well we preprocessed our data.  We were constantly trying different methods out, seeing how they performed, and adjusting based on what we saw.

Steps:

1. Character removal: removed specific characters in our medical transcription column 
2. Word Tokenization : split our transcription sentences into smaller parts (individual words) called tokens
3. POS Tagging : marked up words with specific part of speech, then removed all that were not a form of a noun (nouns are better to use with NLP of medical data)
4. Lemmatized words : converted our tokenized words into its meaningful base form 
5. Dropping Medical Specialties: 








# Summary of Findings






# Visuals







# Challenges







# Limitations 

- Why isn't our model performing even better?

* In medical transcriptions, there is an overlap in the words that are used.  For example, one of our datapoints predicted gastroenterology, however the actual specialty was surgery (could've been surgery of the stomach, so keywords would've overlapped in this exmaple) 

* We had to make sure to not steer our model and overfit it to show our biases.  If we brought too much bias into the process, we would've been taking away the advtange of machine learning.


# Reach Goals (for the future)

- What would make our models better?

* Spending more time analyzing / cleaning text data
* Customize stopwords (would need subject mattter expertise)
* Balancing of the dataset
* More data / medical transcriptions

* Front-end website to input symptoms and correct medical specialty will pop up
