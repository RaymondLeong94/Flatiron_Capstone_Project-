# Using NLP to help Predict and Make Recommendations to for Restaurants Using Adjectives and Nouns

We want to use different Natural language processing techniques in order to understand text data so that we can use similarity scores through our best neural network model to make proper business recomendations for restaurants that have 3.5 stars. This will help isolate the stakeholder and shift them either lower or higher on the scale of stars 1-5.

![image](https://user-images.githubusercontent.com/98904682/218569185-f36debab-8a7c-487a-91c2-05f0f974fdc7.png)
-Image taken from: https://www.architecturaldigest.com/story/philippe-starckdesigned-lavenue-restaurant-opens-at-saks

# Introduction: 

The project is focused on Natural Language Procesing and analyzing text from Yelp's database in order to extract information related to the perfomance of a restuarant (the amount of stars it has). Natural Language processing has five rules obtained from [communication community](https://www.communicationcommunity.com/5-domains-of-language/): Fortunately we can get the syntax, semantics and pragmatics through NLP but we cannot obtain the tone of the voice and this project has potential biases from the reviwer thus influenting the pragmatics. Overall, the following diagram is important because it keeps us centered on the business approach mentioned in notebook 1: How do words influence ratings?

![image](https://user-images.githubusercontent.com/98904682/223903817-f6dd6eed-08a1-4690-87d8-441a44cc3b8e.png)

We will utilize 3 branches of lingustics in our notebooks- morphology, syntax and semantics: 


**Important links**

Notebook 1: https://github.com/RaymondLeong94/Flatiron_Capstone_Project-/blob/main/Notebooks/Notebook1%20EDA.ipynb

Notebook 2: https://github.com/RaymondLeong94/Flatiron_Capstone_Project-/blob/main/Final_Notebook.ipynb

Presentation: https://github.com/RaymondLeong94/Flatiron_Capstone_Project-/blob/main/NLP%20Yelp.pdf

### Data source

Please fill out the form and download the data, then just run the notebooks in the same directory as the json files you have downloaded. Additionally the weights and the model embedding itself can be found in Notebooks as model_88.h5 and word_embeddings.npy at this link: https://github.com/RaymondLeong94/Flatiron_Capstone_Project-/tree/main/Notebooks

Dataset link and description

- Yelp's dataset download: https://www.yelp.com/dataset/download

- Yelp's dataset description: https://www.yelp.com/dataset/documentation/main

*Note you must fill out the forms before you can download the files* 

# Stake Holder and Business Understanding 

### Business problem: 

When COVID occured, Ubereats, Door Dash and Grubhub soared in popularity when restaurants were closed- and now that restaurants are opening back up, there comes a need to review the data from the yelp database to help restaurants opening up. **The main issue is that there were no waiters during covid, thus reintroducing staff has its costs** However, owners are struggling to allocate their resources properly and do not know where to invest in either: their service (now that everyone is coming back to dining) or food quality. 

### Stakeholder

Mark Cuban is a huge investor in Philledelphia restaurants and has an average rating of 3.5 stars across his restaurants.

The job is to solve a classification problem (restaurants higher in 3.5 star rating and also lower), by looking at relevent nouns and adjectives with a certain accuracy to predict whether to invest in food quality or service.


# Data Understanding
The following diagram is the ERD for the json files used in this project

![image](https://user-images.githubusercontent.com/98904682/223905058-c553d9df-e0f7-4bdf-a2e7-cc061bf0c39d.png)

From notebook 1 we had to decide which corpus of text we should use, either tips (abbreviated reviews) or the full reviews themselves. In order to distinguish the corpus we decided that the one that is used needs to have as little ambuguity as possible by having the least amount of neutral text. 

**Thus we decided that reviews was better**

![image](https://user-images.githubusercontent.com/98904682/223907152-e42b5a34-eb49-473b-b7f5-3fd088fa66ba.png)

![image](https://user-images.githubusercontent.com/98904682/223907171-2d283be7-794c-463e-b7f8-aae844d46591.png)

**From notebook 1 we were able to find the most popular state with restaurant reviews:**

![image](https://user-images.githubusercontent.com/98904682/223907505-710a1771-36e3-4890-b940-7b7e1371c8ca.png)

Furthermore, if we used a word 2 vec model we found that the following confusion matrix for a Count Vectorization followed by a MultinomialNB has an accuracy of 73.33 and 66% precision

![image](https://user-images.githubusercontent.com/98904682/223909208-0b282532-8938-4466-9b1b-c810162528fd.png)


# Modeling

Using Named entity recognition we found the following nouns and adjectives:

![image](https://user-images.githubusercontent.com/98904682/223907728-7949639e-d89c-45e2-86d5-bd91e0f73e00.png)

![image](https://user-images.githubusercontent.com/98904682/223907751-0d221090-aba8-4457-810a-dc5654c15f07.png)

Nouns is probably the most important contributor to a review because it tells us what we need to focus on improving. When we look at a word cloud we are visualizing the frequency and what is most talked about. In this case, order is larger than flavor- this means that the word order occurs more often but we cannot deduct that just because it has occured more in the corpus that it is more signifiicant

What we can say is that, in addition to service or food quality, we should also note the price of the restaurant 

We then looked at the similarity scores from our word2vec model in eda

![image](https://user-images.githubusercontent.com/98904682/223908049-c3573266-3502-4cf3-aa01-73aa2336e482.png)

Some example implicit recommendations we can make are:

- As per silverware: Waiters should be more careful with silverware presentation

- As per "request" and "timecrowd" (timecrowd may mean time related to crowds): Waiters should be careful with time management

- Ask per ask: Waiters should be vigilent about the customer's needs because you dont want the customer to constantly ask for the waiter.

With keras we used a vocabulary size of a 1000 to achieve an accuracy of 88% on the test and validation data. We remade our vectorized word df and found the cosine similarity scores after the word embedded weights were used.

![image](https://user-images.githubusercontent.com/98904682/223908512-971c347c-3622-454a-86fd-4add3596d753.png)

With the NN model we see that providing butter to your food, making the place look new by fixing it and having loud servers that can talk over the noise can contribute to a review that helps Mr. Cuban move up.

Looking at negative correlations

![image](https://user-images.githubusercontent.com/98904682/223908599-595618af-5d77-43af-9c1b-b10ad9d9673f.png)

Reasonable servers are not associated with server, which means that waiters should be automatic and stick to a system every time. Tired workers that smoke are negatively correlated with servers, or if they always bunch up. Thus these negative correlations imply traits servers should not have.

![image](https://user-images.githubusercontent.com/98904682/223908670-edc5274c-c4b9-4d3e-919b-95e5d13f87da.png)

It seems that fruits (desert), gravy and other condiments are associated with food. Thus Mr. Cuban should give these out at the lowest cost to save money. Additionally we see worker being associated with food- this may lead to a lot of further research into the matter

# Conclusions:

What we saw in the eda about the sevrer's behavior is important to helping Mr. Cuban grow his business and distinguish himself from 3.5 stars and aim to achieve better. He has a model that is 88% accurate in predicting nouns and adjectives from a review which can highlight important business actions.

## Recommended Business actions:
He will want to hire friendly staff that are different and unique. They should be vigalent on the course of the meal and offer free condiments and desserts so that users can give more positive reviews. This means that they can then can improve Mr. Cuban's overall rating by creating more distinguishment from other 3.5 stars.

He would also have to focus on the servers behaviors, smoking and slacking are negative correlated with server. As from the statements from previously in EDA, there are many different ways to improve service.

Main packages used:
-vaderSentiment, NLTK, WordCloud, Numpy, Pandas, Geo.py, re,  matplotlib, Random forest, SVD, MNB


## Repo Navigation:

├── .gitignore

├── Notebooks 

├──Final_notebook.ipynb

├── NLP Yelp.pdf

├── README.md

└── requirements.txt
