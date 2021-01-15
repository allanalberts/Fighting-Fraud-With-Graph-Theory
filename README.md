# Impoving-Trust-in-Marketplace-User-Ratings
Creating a trusted environment is essential for successful online marketplaces. Unfortunately, unscrupulous users will devise schemes to manipulate user ratings and thus, undermine this trust. My project's goal is to build a predictive model to identify users who are likely to create negative experiences for other marketplace users. 

### Bitcoin Trading Marketplaces
The OTC Bitcoin Marketplace is a place where users can trade bitcoin for service or products or vice versa. For my project I focused on the Over The Counter Bitcoin Marketplace. 

![](/images/OTC_screenshot.png)

 Trust is established by having users rate each other based on the experience of their interactions. Ratings range from -10 for completely untrustworthy to +10 for very trusted. My project focuses on the first five years of operation of this marketplace which involves 5,881 users giving 35,588 ratings to each other, 10% of which were negative. 

![](/images/bitcoin_marketplace_hist.png)

### The Fraud Ratings Manipulation Scheme
Someone trying to defraud a user needs to establish ratings credibility before anyone is likely to engage with them in a transaction of any significant value. They can work to obtain a higher rating by engaging in legitimate transactions, but this takes time and fraudsters are usually not very patient.  Alternatively, they can create a bunch of fake users and then have all of these users rate each other positively. Once the users have a good rating, they go out and find an unsuspecting legitimate user to defraud.

<img src="/images/example1.png" alt="drawing" width="500"/>
<img src="/images/example2.png" alt="drawing" width="500"/>

### Fraud Detection Models
Not only is this fraud difficult to anticipate by the marketplace users, it is also difficult to identify by traditional fraud detection models. Models are usually based on features that measure the velocity and pattern of a user’s activity and learn which patterns are normal and which ones are likely fraudulent. However in this case, we need to look beyond the fraud user and see what is happening with people they have previously interacted with.

### Graph Theory Features
To address this issue, I have engineered 10 predictive features based on graph theory. For each transaction, I used the history of prior positive ratings in the marketplace and generated a Reverse Directed Ego Subgraph of the users rating connections. The subgraph shows interactions between users that the main user has received positive ratings from. This subgraph is then used as the basis for generating predictive features associated with the user, mostly around density metrics. These features consist of:
- Counts of triads, specificaly triads 301, 210, 201, and 120
- Measure of centrality, specifically high closeness and low betweeness to signify a denser network
- Cluster coefficient
- Number of degrees or users that where interacted with
- Number of cliques

### Other Features
My model also uses 12 more traditional features associated with user activity including, negative ratings in previous 24 and 48 hours, number of successive negative ratings, days since last active, first active and last negative rating, number of positive and negative ratings, sum and average of ratings received and negative ratings percent.

### The model:
My fraud detection model utilizes these 22 features with a Random Forest Classifier. I trained the model on 80 percent of the marketplace ratings using a stratified and shuffled sample and tested performance against the remaining 20% of ratings. I used scikit-learn's RandomizedSearchCV method to define a grid of paramaeter ranges and randomly sampled from the grid performing 3-fold cross validation. Next I used a grid search with cross validation for final tuning. Tuning resulted in a 2.64% increase in model performance. The accuracy score of 0.9437 was extemely close to the Out Of Bag score of 0.09462 thus suggesting model validity. My model was able to successfully predict a negative rating 54% of the time with only 1% of legitimate users affected with a false positive prediction when using a 0.5 threshold value. For the fraud ratings scenario described above, I was able to sample the results and determine successful classification after the graph theory features had been added to the model. 

<img src="/images/confusion_matrix.png" alt="drawing" width="500"/>

<img src="/images/PR_curve.png" alt="drawing" width="500"/>

### Next Steps - Results Validation
The model performed slightly better with the graph theory features than without with the f1_score increasing from .65 to .66. As there are relatively few frauds tied to the scenario I am trying to detect, I would like to ensure that this increase is due to them being detected. I plan to run a series of random train-test-splits to test the model with and without the graph features and then compare the results to determine if they are statistically significant, especially for the scenarios I have manually categorized as ratings manipulation fraud. 

Data Source: https://snap.stanford.edu/data/soc-sign-bitcoin-otc.html

OTC Bitcoin Marketplace: https://bitcoin-otc.com