## Result of userCF
### MovieLens
Below is the result for MovieLens data set. The recommended items is not ensured to be the totally new item for the user.(the user may have touched the item in the training data set.) With IIF (add penalty to popular common items when increase user-user similarity) in the cosine distance, the recall and precision increase in all groups. This proves the rationality for IIF.
![avatar](plot/movielens_not_ensure_new.png)