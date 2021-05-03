# Clinical_Notes_Machine_Learning


# Topic Selection

Medical data is extremely hard to find due to HIPAA (Health Insurance Portability and Accountability Act) privacy regulations. As a group, we decided to dip our feet into the fast growing field of Machine Learning and NLP (Natural Language Processing) to extract valuable information from unstructed medical text data. 
Why NLP in healthcare? 
A large part of medical information is reported as free-text and patient clinical history has been conveyed for centuries by all medical professionals in the form of notes, reports, trascriptions. 

# Project overview

For our final project, our group chose to use a dataset (from Kaggle, [Kaggle](https://www.kaggle.com/tboyle10/medicaltranscriptions?select=mtsamples.csv)) that contained medical transcriptions and the respective medical specialties (4999 datapoints). We chose to implement multiple supervised classification machine learning models - after heavily working on the corpora - to see if we were able to correctly classify the medical specialty based on the trascription text. 

# Dataset

[Kaggle](https://www.kaggle.com/tboyle10/medicaltranscriptions?select=mtsamples.csv)

![Image](https://github.com/AliceSartori/Medical_Specialist_Machine_Learning/blob/main/plots/original_dataset.png)


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
* GridSearch
* TfidfVectorizer
* Yellowbrick

# Pipeline

**Preprocessing Data**

Our project has three notebooks which shows the different phases of our work as far as cleaning, ideas, and analysis. 
Our method was a constant give and take - trying different methods, seeing how they perfomed, and adjusting based on what we saw.

*Notebook 1*

1. Character removal: removed specific characters in our medical transcription column 
2. Word Tokenization : split our transcription sentences into smaller parts (individual words) called tokens
3. Stopwords: removed 'stopwords' (words that provide no meaning to sentence)
3. Dropped / Combined Medical Specialties: removed specialties not needed (workers comp, hospice, etc.) / combined specialties together
4. Lemmatized words : converted our tokenized words into its meaningful base form
5. POS Tagging: marked up words with specific part of speech


*Notebook 2*

6. Performed POS Tagging before lemmatizing the words
7. Removed all words that were not nouns or a form of a noun (nouns contain the meaning)


*Notebook 3*

8. Reduced medical specialties : there was a major imbalance in our dataset, so we wanted to reduce the amount of the higher categories, and combine some of the lower categories into 'other specialties'

**Creating the Model**

1. Converting Text to Word Frequency Vectors
2. Used RandomForest & NaiveBayes


# Transcription Data after all preprocessing


![Image](https://github.com/AliceSartori/Medical_Specialist_Machine_Learning/blob/main/plots/preprocessed_dataset.png)


# Token / Lemmatization Graphs

![Image](https://github.com/AliceSartori/Medical_Specialist_Machine_Learning/blob/main/plots/Corpus_view_with_tokens_number_WITHOUTREDUCTION.png)
                    Tokens # (Without Reduction)

![Image](https://github.com/AliceSartori/Medical_Specialist_Machine_Learning/blob/main/plots/Corpus_view_with_Lemmas_after_first_reduction.png)
                    Lemmatized # (With Reduction)

![Image](https://github.com/AliceSartori/Medical_Specialist_Machine_Learning/blob/main/plots/tokens_plot_total_corpus.png)
                                POS Tags



# Results



# Summary of Findings Graphs

![Image](https://github.com/AliceSartori/Medical_Specialist_Machine_Learning/blob/main/plots/classification_report_GRIDSEARCH_WITHREDUCTION.png)
                    Classification Report (gridsearch with reduction)

![Image](https://github.com/AliceSartori/Medical_Specialist_Machine_Learning/blob/main/plots/classification_report_RANDOMFOREST_FIRST_ANALYSIS.png)
                    Classification Report (randomforest without reduction)

![Image](https://github.com/AliceSartori/Medical_Specialist_Machine_Learning/blob/main/plots/classification_report_RANDOMFOREST_WITHREDUCTION.png)
                    Classification Report (randomforest with reduction)

![Image](https://github.com/AliceSartori/Medical_Specialist_Machine_Learning/blob/main/plots/classification_report_multinomial_FIRSTANALYSIS.png)
                    Classification Report (multinomial without reduction)

![Image](https://github.com/AliceSartori/Medical_Specialist_Machine_Learning/blob/main/plots/classification_report_multinomial_WITHREDUCTION.png)
                    Classification Report (multinomial with reduction)

![Image](https://github.com/AliceSartori/Medical_Specialist_Machine_Learning/blob/main/plots/confusion_matrix_without_reduction.png)
                    Confusion Matrix (without reduction)


![Image](https://github.com/AliceSartori/Medical_Specialist_Machine_Learning/blob/main/plots/confusion_matrix_with_reduction.png)
                    Confusion Matrix (with reduction)


# Main Challenges

* Preprocessing the data: lemmatizing, pos tagging, tokenizing
* Making the dataset more balanced: finding the right amount to remove/keep/adjust


# Limitations 

**Why isn't our model performing even better?**

* Medical data in general is a lot harder to analyze / preprocess as it is very complicated
    
    1. In medical transcriptions, there is an overlap in the words that are used.  For example, one of our datapoints predicted gastroenterology, however the actual specialty was surgery (could've been surgery of the stomach, so keywords would've overlapped in this example) 

    2. Some english stopwords might be useful that were removed ("back", "have", "had")

    3. Some medical stopwords could be removed that weren't ("patient","doctor")

* We had to make sure to not steer our model and overfit it to show our biases.  If we brought too much bias into the process, we would've been taking away the advtange of machine learning.

* Lack of medical data information


# Reflection

**What would make our models better?**

* Spending more time analyzing / cleaning text data
* Customize stopwords (would need subject mattter expertise)
* Balancing of the dataset
* Finding more data / medical transcriptions


