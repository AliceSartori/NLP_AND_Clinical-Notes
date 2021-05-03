# NLP & Clinical Notes


# Topic Selection

Medical data is extremely hard to find due to HIPAA (Health Insurance Portability and Accountability Act) privacy regulations. As a group, we decided to dip our feet into the fast growing field of Machine Learning and NLP (Natural Language Processing) to extract valuable information from unstructed medical text data. 
Why NLP in healthcare? 
A large part of medical information is reported as free-text and patient clinical history has been conveyed for centuries by all medical professionals in the form of notes, reports, trascriptions. 

# Project overview

For our final project, our group chose to use a dataset (from [Kaggle](https://www.kaggle.com/tboyle10/medicaltranscriptions?select=mtsamples.csv)) that contained medical transcriptions and the respective medical specialties (4999 datapoints). We chose to implement multiple supervised classification machine learning models - after heavily working on the corpora - to see if we were able to correctly classify the medical specialty based on the trascription text. 

# Tools / Technology Used

* CSV File                              
* Pandas                                
* Numpy                                 
* Matplotlib                            
* NLTK (Natural Language Toolkit)
* RandomForest
* Multinomial Naive Bayes
* Scikit-learn
* LogisticRegression
* Hyperparameter tuning with GridSearchCV (RandomForest)
* TfidfVectorizer
* Yellowbrick

# Pipeline

**Preprocessing Data**

Our project has three notebooks which shows the different phases of our work as far as cleaning, ideas, and analysis. 
Our method was a constant give and take - trying different methods, seeing how they perfomed, and adjusting based on what we saw.
Across all notebooks cleaning the data set and getting rid of characters in the raw text even before the tokenization.
![Image](https://github.com/AliceSartori/NLP_AND_Clinical-Notes/blob/main/Screen%20Shot%202021-05-03%20at%2012.33.12%20PM.png)

*Notebook 1 Classic Approach*
1. Character lowercase and removal: removed specific characters in our medical transcription column 
2. Word Tokenization: split our transcription sentences into smaller parts (individual words) called tokens
3. Stopwords: removed 'stopwords' (words that provide no meaning to sentence)
3. Dropped / Combined Medical Specialties: removed specialties not needed (workers comp, hospice, etc.) / combined specialties together
4. Lemmatized words : converted our tokenized words into its meaningful base form
5. Limit the lenght of our corpora
6. POS Tagging: marked up words with specific part of speech

*Notebook 2 Medical Approach*
From Notebook 2, we gave more importance to Nouns on the assumptions that the Medical vocabolary Nouns are specific to this field and they carry the true meaning. We also were more agressive and removed dates, numbers, PHI etc.
1. We eliminated more characters 
2. Performed POS Tagging before lemmatizing the words to remove all POS tags that were not Nouns
3. 4. Didn't take away English Stop Words (ex. English words like 'back' are treated as stop words, but in medical those field might be useful)

![Image](https://github.com/AliceSartori/Medical_Specialist_Machine_Learning/blob/main/plots/tokens_plot_total_corpus.png)


*Notebook 3 Resampling*

Machine Learning algorithms tend to produce unsatisfactory classifiers when faced with imbalanced datasets. Standard classifier algorithms like Decision Tree and Logistic Regression have a bias towards classes which have number of instances. They tend to only predict the majority class data. The features of the minority class are treated as noise and are often ignored. Thus, there is a high probability of misclassification of the minority class as compared to the majority class.
Our data set had 2 numerous classes (Surgery and Consultation) that were predicted more than the others.
We reduced their amount of datapoint of these two categories with a sampling function to keep the random factor and combine some of the categories with lower amount od datapoints into 'Others'.

**Creating the Model**

1. Converting Text to Word Frequency Vectors: there are several ways to do this, such as using CountVectorizer and HashingVectorizer, but the TfidfVectorizer is the most popular (how many times a term appears in a document/reciprocal of number of times a term appears in all documents)
2. From this conversion: Measure of how informative a term  (occurrence of rare term is more informative than that of a widely used term/terms used frequently in a document are more informative that terms used only once)
3. Algorithms used: NaiveBayes (Multinomial), RandomForest, Hyperparameter tuning with GridSearchCV (RandomForest), Logistic Regression.


# Transcription Data after all preprocessing


![Image](https://github.com/AliceSartori/Medical_Specialist_Machine_Learning/blob/main/plots/preprocessed_dataset.png)


# Token / Lemmatization Graphs

![Image](https://github.com/AliceSartori/NLP_AND_Clinical-Notes/blob/main/plots/Corpus_view_with_tokens_number_WITHOUTREDUCTION_Beginning.png)
                    Tokens #: after some reduction in specialties (8) and before deciding to put a lowerbound and upperbound for their #

![Image](https://github.com/AliceSartori/Medical_Specialist_Machine_Learning/blob/main/plots/Corpus_view_with_Lemmas_after_first_reduction.png)
                    Lemmatized # (With Reduction, before deciding to reduce Surgery and Consultation )


# Results


# Summary of Findings Graphs
Precision can be seen as a measure of a classifier’s exactness. Said another way, “for all instances classified positive, what percent was correct?”

Recall is a measure of the classifier’s completeness; the ability of a classifier to correctly find all positive instances. Said another way, “for all instances that were actually positive, what percent was classified correctly?”

F1 score is a weighted harmonic mean of precision and recall such that the best score is 1.0 and the worst is 0.0. Generally speaking, F1 scores are lower than accuracy measures as they embed precision and recall into their computation. As a rule of thumb, the weighted average of F1 should be used to compare classifier models, not global accuracy.

Support is the number of actual occurrences of the class in the specified dataset. Imbalanced support in the training data may indicate structural weaknesses in the reported scores of the classifier and could indicate the need for stratified sampling or rebalancing. 

As it can be easily seen on the classification report heatmaps, Hyperparameter tuning with GridSearchCV on the RandomForest gave us the best result. 
Note that for Dentistry, we had very good results before blending it to the category "Others", most probably due to the very sectorial vocabulary of a dentist.

## MultinomialNB 

![Image](https://github.com/AliceSartori/Medical_Specialist_Machine_Learning/blob/main/plots/classification_report_multinomial_FIRSTANALYSIS.png)
                 

![Image](https://github.com/AliceSartori/Medical_Specialist_Machine_Learning/blob/main/plots/classification_report_multinomial_WITHREDUCTION.png)
                  

## RandomForest
![Image](https://github.com/AliceSartori/Medical_Specialist_Machine_Learning/blob/main/plots/classification_report_RANDOMFOREST_FIRST_ANALYSIS.png)
                 
                    
 ![Image](https://github.com/AliceSartori/Medical_Specialist_Machine_Learning/blob/main/plots/classification_report_RANDOMFOREST_WITHREDUCTION.png)
                       
                    
![Image](https://github.com/AliceSartori/Medical_Specialist_Machine_Learning/blob/main/plots/classification_report_GRIDSEARCH_WITHREDUCTION.png)
                   


![Image](https://github.com/AliceSartori/Medical_Specialist_Machine_Learning/blob/main/plots/confusion_matrix_without_reduction.png)
                    Confusion Matrix (without reduction)


![Image](https://github.com/AliceSartori/Medical_Specialist_Machine_Learning/blob/main/plots/confusion_matrix_with_reduction.png)
                    Confusion Matrix (with reduction)


# Main Challenges

* Preprocessing the data: cleaning, lemmatizing, POS tagging, tokenizing
* Making the dataset more balanced: finding the right amount to remove/keep/adjust
* Finding the algorithm to use


# Limitations 

**Why isn't our model performing even better?**

* Medical data in general is a lot harder to analyze / preprocess as it is very complicated
    
    1. In medical transcriptions, there is an overlap in the words that are used.  For example, one of our datapoints predicted gastroenterology, however the actual specialty was surgery (could've been surgery of the stomach, so keywords would've overlapped in this example) 
    
    2. In clinical notes the same text could be repeated by the same practitioner (a lot of copy and paste)

    3. Some medical stopwords could have been removed ("patient", "doctor", "diagnosis") due to a high/high relationship TF/IDF

* We had to make sure to not steer our model and overfit it to show our biases.  If we brought too much bias into the process, we would've been taking away the advatange of machine learning.

* Lack of medical data information


# Reflection

**What would make our models better?**
* Short term: Random Search Cross Validation using  RandomizedSearchCV method: we can define a grid of hyperparameter ranges and randomly sample from the grid. On each iteration, the algorithm will choose a different combination of the features. However, the benefit of a random search is that we are not trying every combination, but selecting at random to sample a wide range of values. Random search will allow us to narrow down the range for each hyperparameter. After that, we will know where to concentrate our search and we will be able to explicitly specify every combination of settings to try. We can do this with GridSearchCV.
* Short term: Customize stopwords 
* Spending more time analyzing / cleaning text data (would need subject matter expertise)
* Balancing of the dataset
* Finding more data / medical transcriptions for the specialties with less datapoints


