# Impoving-Trust-in-Marketplace-User-Ratings
This project utilizes graph theory and deep learning to identify fraud in P2P marketplace user ratings.


The ability to trust other users is a key requirement for successful peer-to-peer (P2P) marketplaces. Trust is established through user ratings of interactions with other users. However, ratings can be manipulate by by sellers who use multiple accounts that they control to create fake positive ratings for themselves. This project combines graph theory with deep learing to identify manipulated user ratings.

### Bitcoin Trading Marketplaces
Bit coin P2P markplaces are particularly suseptable to fraud involving manipulated user ratings. (talk about why this is) 
- explain the marketplaces and show timeline overview of ratings activity

### Idendifying Malicious Bot Activity
Fraudsters often use malicious bots or automated programs to perform their dirty work. Use of these programs usually manifests itself as a data anomaly amongst human activity. In the case of ratings, the bot activity appears as high frequency ratings in short bursts. The first step in countering frauding threats is to identify and remove ratings that are generated automatically. 
- go into anomalous detection of automated programs. 
- show eda graph of bot activity

### Identifying Ratings Boosted through User Colusion
- explain how fraud occurs
- show eda example visualization of fraud

### Engineering Features to identify User Colusion
- go into how node2vec works

### Testing Strategy
### Predict ratings fraud using Neural Net
### Create scoring system based on predictions
### Review predition outcomes vs use of avg ratings
- confusion matrics
- roc plot


