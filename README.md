# #Using Yelp's dataset to help PA restaurants identify areas of improvement in Service or Food Quality. 

## How?
Using Yelp's dataset we can find entities that are of nouns or adjectives through NLTK, from there we can use a word2vec model to create a keras layer with 76% accuracy. This accuracy reflects the words used are representative of the output variable "stars". This embedding layer is important for future training purposes. In the present it gives us words related to service or food quality that resturants in PA can focus on in order to recieve higher stars on their establishments and increase revenue. 

![image](https://user-images.githubusercontent.com/98904682/218569185-f36debab-8a7c-487a-91c2-05f0f974fdc7.png)
-Image taken from: https://www.architecturaldigest.com/story/philippe-starckdesigned-lavenue-restaurant-opens-at-saks

## Business Understanding 
- Using Yelp, can we predict meaningful words associated in reviews to help improve hospitality or food service in the yelp’s database?

- Do we use an abbreviated review or a full review?

- How can we translate these findings into data driven actions?

- According to Dylan from 'https://qsrautomations.com/blog/restaurant-management/restaurant-customer-service/', his five points of improve service and hospitality 

- Clear staff expectations, personalized experiences, streamline wait times, response to concern, do table touches. These are all aspects and words which we have already hit on.

## Data: All five files can be found at: https://www.yelp.com/dataset/
- There are five files but only business.json, review.json, tips.json are relevant for this project .

-  Due to machine limits, only 600,000 reviews were pulled from reviews.json

- There are 150243 unique businesses, with 64,616 food establishments with 67,000 reviews for 7076 restaurants

Modeling
Count Vec
![image](https://user-images.githubusercontent.com/98904682/218570824-c328e6d2-1238-43c7-9aa9-6cfd92efd265.png)

N_gram word vec
![image](https://user-images.githubusercontent.com/98904682/218571056-f3cc807b-9be3-443b-9726-dc17568fc222.png)

Service Recommendations
- Waiter and Waitress show similar cosine similarity scores for certain behaviors.
- For instance refilling is commonly associated with the job, so a business recommendation may be to advise your staff to walk around with water when busy.
- Another task is probably taking orders from customers, don’t overcomplicate things or arrive too early before they’ve made up their mind on their order.

Food Recommendations
-The words; balance and texture appear most similar with flavor.

- Some areas of flavor to explore might be spice and sweet flavors that have ‘hints’ of other ‘combinations’

- Additionally, consistency with flavor appears to high in similarity scores.


Main packages used:
-vaderSentiment, NLTK, WordCloud, Numpy, Pandas, Geo.py, re,  matplotlib, Random forest, SVD, MNB
