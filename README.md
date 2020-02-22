# Models For Top-n Recommendation

## Project structure
```
- ./
  -base
    --Item.py (class for item)
    --Model.py (base class for all models)
    --Tag.py (class for tag)
    --User.py (class for user)
  -data
    (data sets used)
  -models
    -model_struc
      (keras model structure plots)
    -saved_models
      (saved pickle and .h5 model files)
    --*.py (implement of different models)
  -utils
    --Data_util.py (util for data processing)
    --Feature_util.py (util for feature engineering)
  --run_model.py (run different models from here)
  --evaluate_model.py (evaluate different models)
```

## Model types
1. user collabrative filtering -> UserCF.py
2. item collabrative filtering -> ItemCF.py
3. latent factor model -> LMF.py
4. simple tag based model -> TagBasic.py
5. wide and deep -> Wide_and_deep.py
6. most popular -> Popular.py
7. random model -> Random.py

## Some details
### Timestamp
In data processing period, user's events is sorted by timestamp. Then for one user, its event series is divided and push into training set and test set. This makes sure that, for one user, its events in test set is later than those in training set.

In UserCF and ItemCF, if argument `timestamp` is set to `True`, then time elapse is considered when computing the similarity score.

For UserCF, during user-user similarity computing period, common items of two users with close timestamp will add more contibution to their similarity. During potential items rank period, item from similar user with timestamp closer to current timestamp will have higher score.

For ItemCF, during item-item similarity computing period, two items from one user's history list will have higher similarity if their timestamp at which the user touched are close. During potential items rank period, similar items of one user's more recent history item will have higher score.

### Penalty for popularity
For UserCF, penalty of item's popularity is considered. If a common item between two users is very popular, this item will contribute less to the similarity of these two users.

For ItemCF, penalty of user's popularity is considered. If two items are touched by a user, whose history items list is very large, then this user will contribute less to the similarity of these two items.

For tag based model, both item's and user's popularity are considered.

### Standardization
For UserCF, user-user's similarity score is standardized by the number of their history items. Consider user A with item (a, b, c), user B with item (a, b, c, ..., z), user C with item (a, b, c). If not standardize, these three users' similarities are same. Which is not true, because user A and C are definitely more similar.

For ItemCF, item-item's similarity score is standardized by the items' popularity. Consider item A is a classic action movie covers almost everyone, item B is classic romance movie covers almost everyonbe. If we don't standardize by the number of their covered users, these two movie's similarity will be very high, which is not the case.

### Normalization
For ItemCF, during the potential items rank period, for one history item and its K most similar items with similarities, these similarities are normalized so that they sum to 1. This procudure avoid the high similarities of corresponding K items introduced by a popular history item.

### Negative samples
For LFM and Wide&deep model, negative samples for each user is created. Parameter `neg_frac` refers to the ratio of negative samples size over positive samples size. In the raw event data, every user record is a positive sample for the user. The negative samples are created by using popular items that are not touched by the user. See function `create_negative_samples` in `Data_util.py` for more details.

### Wide&deep model
For movie category feature crossing,  we consider all the 19 categories to be crossed together. Theoretically, the number of all possible crossing values is 2^19, which will results in an embedding table of size (2^19)*dim. However, there are some possible values that are not going to show in real world. For example, a movie of both child and horror categories. So we can keep some extent of feature crossing's diversity rather than take all possible crossing values into account. This is defined in `hash_bucket_size` of `tf.feature_column.crossed_column`.

#### Embedding
The embedded vectors of users and items can also be extracted by user/item id to show the similarity of users or items. The more closer two users' vector are (with a distance metric), the more similar they will be.

## Evaluation
Recall and fallout is used to produce ROC curve and compute partial AUC for each model with different n (number of recommended items).

In top-n recommendation senario, one user's touched item list in test set is `items_real`. Model generated recommend item list for this user is `items_reco`. The common item list of `items_real` and `items_reco` is `items_hit`. Recall is computed by `n_TP/len(items_real)`. Precision is computed by `n_TP/len(items_reco)`. Fallout is computed `n_FP/(n_all_items-len(items_real))`. As in other classification senario, there is a trade off between recall and precision. If we recommend all items to the user, the recall will be 100% while precision will be very low. If we recommend only one most relevant item, the precision will much likely to be 100% or very high, while recall for avtive users will be very low. To evaluate model's performance, many recall-fallout pairs is computed under different n values. Then the partial ROC curve and AUC is evaluated. 

Note that for UserCF, ItemCF and TagBasic model with fixed k (number of similar objects to consider) smaller than a certain number, they may not be able to generate a recommend items list of a rather large length.