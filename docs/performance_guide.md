### Speed Tips:

- We recommend setting  thread number equal to cpu_cores_number. If your machines have enough memory and cpu cores,  use as less machines as possible to reduce network communication.
- If the task contains lots of categorical features, and  most of the features are sparse, you can filter the features whose frequency of occurrence is less than "feature.filter_threshold", or provide a custom feature dictionary  via setting "model.dict_path" and "model.need_dict".
- If your feature dim is huge, training will procede very slowly. You can use feature hash to reduce dim at a fraction of cost via setting "feature.feature_hash" except for gbdt. 
- If the classification task has serious imbalanced dataset，you can set "data.y_sampling" in model configuration to reduce the number of samples. For instance, CTR prediction scene usually has large proportion of negative samples，“y_sampling : [0@0.1]” means that it reserves 10% of negative samples randomly, and those reserved negative samples enlarge 10x weight as compensation.
- Complex models(fm, ffm, gbst) have more parameters to optimize,  so before training complex models, you can use linear model with l1 regularization to perform feature selection, set "model.dict_path" with linear model dict(saved in the same directory with model file).
- In GBDT, If  the number features is large, use feature-parallel training.  If the number of data is large, use data-parallel training. In mose cases, when you use data-parallel training, the smaller the number of feature bins is, the faster the training process will be, meanwhile, the model will be less accurate.

### Model Accuracy

- The nonlinear ability of models in ytk-learn: gbdt > gbhmlr > gbhsdt > gbmlr > gbsdt > ffm > fm > linear, but use simpler model as much as possible if it meets your need.
- Many learning algorithms(e.g. l1, l2 regularization) assume that features are centered around zero and have variance in the same order, so using feature transformation wil help via setting "feature.transform" except for gbdt.
- Generally speaking,  if the feature number is less than 1000, use gbdt as much as possible.

### Control Overfitting

Use "tail -f log/master.log | grep "train loss" and "tail -f log/master.log | grep "test loss" to monitor whether or not your model is overfitting or underfitting. If your model is overfitting, here are some tips for you:

- adding regularization is a powerful method to avoid over-fitting. Ytk-learn provides grid search and hoag two hyperparameter optimization methods except for gbdt. If your model is convex, using hoag can find optimal hyper parameters faster.
- early stopping
  - decease "optimisation.line_search.lbfgs.convergence.max_iter" value.
  - increase "optimisation.line_search.lbfgs.convergence.eps" value.
- increase regularizations: increase "loss.regularization.l1/l2" value.
- decease model complexity:
  - fm/ffm: reduce latent factor size("k").
  - gbst
    - reduce mixture number("k").
    - add randomness to make training robust("subsample", "feature_sample_rate").
  - gbdt
    - reduce tree max depth("optimization.max_depth").
    - decrease maximum leaf count("optimization.max_leaf_cnt").
    - increase minimum hessian sum in a leaf("optimization.min_child_hessian_sum").
    - increase minimum number of samples to make a split("optimization.min_split_samples").
    - add randomness to make training robust("instance_sample_rate", "feature_sample_rate").
