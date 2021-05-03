# Medical_Specialist_Machine_Learning


# Topic Selection


Machine learning is an area that is seeing more acceptance and use in the healthcare industry.  There are many technological advances in handling and analyzing medical data, as it is very clear that it can provide meaningful advances to improve healthcare and therefore the well being of people.  

In specific, machine learning algorithms in Natural Language Processing (NLP) are being widely used in Healthcare & Life Sciences to extract information from unstructured text data.  The ultimate goal of NLP is to read, understand, and make sense of languages in a valuable way.  

As a group, we decided that we wanted to dip our feet into this fast growing field and apply our machine learning skills to see what we could accomplish!


# Project overview

For our final project, our group chose to use a dataset (from Kaggle) that contained medical transcriptions and the respective medical specialty for around 5000 patients. We chose to implement a supervised machine learning model for Natural Language Processing (NLP) that would predict the medical specialty to which each patient needs based on the medical transcriptions.

# Dataset

[Kaggle](https://www.kaggle.com/tboyle10/medicaltranscriptions?select=mtsamples.csv)

![Image](https://github.com/AliceSartori/Medical_Specialist_Machine_Learning/blob/main/plots/original_dataset.png | width=100)


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
* TfidfVectorizer

# Process

*Preprocessing Data*

Our project has three notebooks which shows the different methods we took when cleaning our data.  Our method was a constant give and take - trying different methods, seeing how they perfomed, and adjusting based on what we saw.

Notebook 1 

1. Character removal: removed specific characters in our medical transcription column 
2. Word Tokenization : split our transcription sentences into smaller parts (individual words) called tokens
3. Stopwords: removed 'stopwords' (words that provide no meaning to sentence)
3. Dropped / Combined Medical Specialties: removed specialties not needed (workers comp, hospice, etc.) / combined specialties together
4. Lemmatized words : converted our tokenized words into its meaningful base form
5. POS Tagging: marked up words with specific part of speech

After running notebook 1 and with further research, we changed a few things:

Notebook 2

6. Performed POS Tagging before lemmatizing the words
7. Removed all words that were not nouns or a form of a noun (nouns contain the meaning)

After running notebook 2 and seeing our results, we changed something else:

Notebook 3

8. Reduced medical specialties : there was a major imbalance in our dataset, so we wanted to reduce the amount of the higher categories, and combine some of the lower categories into 'other specialties'

*Creating the Model*

1. Converting Text to Word Frequency Vectors
2. Used RandomForest & NaiveBayes

# Data after preprocessing



# Token / Lemmatization Graphs

![Image](https://github.com/AliceSartori/Medical_Specialist_Machine_Learning/blob/main/plots/Corpus_view_with_tokens_number_WITHOUTREDUCTION.png | width=100)

![Image](https://github.com/AliceSartori/Medical_Specialist_Machine_Learning/blob/main/plots/Corpus_view_with_Lemmas_after_first_reduction.png | width=100)

![Image](https://github.com/AliceSartori/Medical_Specialist_Machine_Learning/blob/main/plots/tokens_plot_total_corpus.png | width=100)



# Results



# Summary of Findings Graphs

![Image](https://github.com/AliceSartori/Medical_Specialist_Machine_Learning/blob/main/plots/classification_report_RANDOMFOREST_FIRST_ANALYSIS.png | width=100)

![Image](https://github.com/AliceSartori/Medical_Specialist_Machine_Learning/blob/main/plots/classification_report_RANDOMFOREST_WITHREDUCTION.png | width=100)

![Image](https://github.com/AliceSartori/Medical_Specialist_Machine_Learning/blob/main/plots/classification_report_multinomial_FIRSTANALYSIS.png | width=200)

![Image](https://github.com/AliceSartori/Medical_Specialist_Machine_Learning/blob/main/plots/classification_report_multinomial_WITHREDUCTION.png | width=100)

![Image](https://github.com/AliceSartori/Medical_Specialist_Machine_Learning/blob/main/plots/confusion_matrix_with_reduction.png | width=100)


![Image](https://github.com/AliceSartori/Medical_Specialist_Machine_Learning/blob/main/plots/confusion_matrix_without_reduction.png | width=100)



# Main Challenges

* Preprocessing the data: lemmatizing, pos tagging, tokenizing
* Making the dataset more balanced: finding the right amount to remove/keep/adjust



# Limitations 

Why isn't our model performing even better?

* Medical data in general is a lot harder to analyze / preprocess as it is very complicated
    
    1. In medical transcriptions, there is an overlap in the words that are used.  For example, one of our datapoints predicted gastroenterology, however the actual specialty was surgery (could've been surgery of the stomach, so keywords would've overlapped in this example) 

    2. Some english stopwords might be useful that were removed ("back", "have", "had")

    3. Some medical stopwords could be removed that weren't ("patient","doctor")

* We had to make sure to not steer our model and overfit it to show our biases.  If we brought too much bias into the process, we would've been taking away the advtange of machine learning.

* Lack of medical data information


# Reflection

What would make our models better?

* Spending more time analyzing / cleaning text data
* Customize stopwords (would need subject mattter expertise)
* Balancing of the dataset
* Finding more data / medical transcriptions


