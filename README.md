# Fake-Product-Review-Montering
The scope and need of online markets and e-commerce platforms are on the rise and many people buy products from these platforms. The amount of feedbacks for products as a result are also present in detail for users to analyze the product they are buying. This can work against the users as well because users can sometime bombard the review section with extreme opinion comments which can work in favor or against the product. Thus, we need to take care of this because this can be done either by the merchant to increase the value of his product or the user to degrade the ratings of that product. 


<b>Features Used: </b>

<ul>
  <li>Sentimental Analysis</li>
  <li>Content Similarity</li>
  <li>Latent Symantic analysis (LSA)</li>
</ul>

## Sentimental Analysis
Sentimental Analysis is contextual mining of text which identifies and extracts subjective information in source material and helping a business to understand the social sentiment of their brand, product or service while monitoring online conversations.
<pre>

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(max_features=2000,min_df =3 ,max_df = 0.6, stop_words = stopwords.words("english"))    
X = vectorizer.fit_transform(corpus).toarray()

#Cretaing TF-IDF from BOW
from sklearn.feature_extraction.text import TfidfTransformer

transformer = TfidfTransformer()
X = transformer.fit_transform(X).toarray()

#Spliting for testing and training

from sklearn.model_selection import train_test_split

text_train,text_test,sent_train,sent_test = train_test_split(X,y,test_size=0,random_state=0)
# here text size = 0 , so that all the data will be used for the training purpose only

# Training our classifier
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(text_train,sent_train)

</pre>

## Latent symantic Analtysis

Latent semantic analysis (LSA) is a technique in natural language processing, in particular distributional semantics, of analyzing relationships between a set of documents and the terms they contain by producing a set of concepts related to the documents and terms. A matrix containing word counts per paragraph (rows represent unique words and columns represent each paragraph) is constructed from a large piece of text and a mathematical technique called singular value decomposition (SVD) is used to reduce the number of rows while preserving the similarity structure among columns Values close to 1 represent very similar words while values close to 0 represent very dissimilar words. 

<pre>
# Latent symantic analysis
# it will analyse all reviews and determine all reviews belong to the same concept
def LSA(text):
    #text is list of reviews of same product
    
    # Created TF-IDF Model
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(text)
    
    # Created SVD(Singular Value Decomposition)
    lsa = TruncatedSVD(n_components = 1,n_iter = 100)
    lsa.fit(X)
    
    terms = vectorizer.get_feature_names()
    concept_words={}
    for j,comp in enumerate(lsa.components_):
        componentTerms = zip(terms,comp)
        sortedTerms = sorted(componentTerms,key=lambda x:x[1],reverse=True)
        sortedTerms = sortedTerms[:10]
        concept_words[str(j)] = sortedTerms
     
    sentence_scores = []
    for key in concept_words.keys():
        for sentence in text:
            words = nltk.word_tokenize(sentence)
            scores = 0
            for word in words:
                for word_with_scores in concept_words[key]:
                    if word == word_with_scores[0]:
                        scores += word_with_scores[1]
            sentence_scores.append(scores)
    return sentence_scores
</pre>

## Content Similarity
With cosine similarity, we need to convert sentences into vectors.Difference in the angle of these determines the similarity between two reviews.
<pre>
from nltk.corpus import stopwords from sklearn.feature_extraction.text import TfidfVectorizer from sklearn.metrics.pairwise import cosine_similarity

tfidf_vectorizer = TfidfVectorizer()
for i in range(len(dataset)):
    
    reviews = [str(dataset["review_body"][i])]
    
    tfidf_vectorizer.fit_transform(reviews)

tfidf_matrix = tfidf_vectorizer.fit_transform(reviews)
    
    #creates TF-IDF Model
    tfidf_list = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix).tolist()
    # Creates matrix based on document similarity
         
    # To check similarity b/w 2 reviews 
    for k in range(1,len(tfidf_list[0])):
                
        if(tfidf_list[0][k]>0.6):
            # 0.6 is defind for the simmilarity level
            
            remove_reviews.append(dataset["review_id"][i+k])
            # i+k is to get the review id of the review

</pre>

# Methods used to determine Fake Reviews
<ol>
<li>Reviews which have dual view </li>
<li>Reviews in which same user promoting or demoting a particular brand </li>
<li>Reviews in which person from same IP Address promoting or demoting a particular brand </li>
<li>Reviews which are posted as flood by same user all the reviews are either positive or negative. </li>
<li>Reviews which are posted as flood by same person from same IP Address </li>
<li>Similar reviews posted in the same time interval </li>
<li>Reviews in which Reviewer using arming tone to by the product <li>Reviews in which reviewer is writing his own story </li>
<li>Meaningless Texts in reviews</li>
</ol>
  
## Future Scope
Finding the opinion spam from huge amount of unstructured data has become an important research problem. Now business organizations, specialists and academics are putting forward their efforts and ideas to find the best system for opinion spam analysis. Although, some of the algorithms have been used in opinion spam analysis gives good results, but still no algorithm can resolve all the challenges and difficulties faced by today’s generation. More future work and knowledge is needed on further improving the performance of the opinion spam analysis.In the future we will do further investigate different kinds of features to make more accurate predictions.

## Prerequisite for this Project
Required pickle files can be found here
https://github.com/anubhavs11/Sentimental-Analysis-using-Logistic-Regression/tree/master/preserved%20files
