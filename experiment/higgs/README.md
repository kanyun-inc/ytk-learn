## HIGGS

Higgs is a binary classification task. The following steps train a GBDT model by **data-parallel( histogram approximate) **algorithm and use this model to get predict results.

1. Download data and generate train and test data

   ```
   # stay in current path ("ytk-learn/experiment/higgs")
   sh get_data.sh
   ```
   You will find "higgs.train" and "higgs.test" in experiment/higgs

2. Train model

   ```
   cd ../.. (go to path "ytk-learn")
   sh experiment/higgs/local_optimizer.sh
   ```

   Training configurations are in ``experiment/higgs/local_gbdt.conf``. Feature bin generating algorithm is configuarated to `` {cols: "default", type: "sample_by_quantile", max_cnt: 255, use_sample_weight: false, alpha: 0.5}``, which means using percentiles to generate bins for all features and max bin cnt is 255.  All samples are equal-weighted, so ``use_sample_weight`` is useless.  ``alpha`` is set to 0.5 which decreases the importance of counts for a specified feature value(compared to 1.0).
   Monitor train info in directory "ytk-learn/log".

   ```tail -f log/master.log``` 

   Training is finished after you see "exit code:0" in log/master.log

   Mode and feature importance files are saved in "experiment/higgs" which is configured in  "local_gbdt.conf"

   Notes：The results of each training may be different because of the unstable algorithm of generating percentiles(feature bins) for each feature.

3. Get predictions  for samples in  "higgs.test"

   ```
   # stay in current path ("ytk-learn")
   sh experiment/higgs/predict.sh
   ```

   Get predict info(evaluation) in directory "ytk-learn/log"

   ```tail -f log/info.log```

   Predicting is finished after you see "predict complete!" in log/info.log, results are saved in experiment/higgs/higgs.test_gbdt_LABEL_AND_PREDICT

   **Notes**: If you want to get the leaf index of each sample, just set ``predict_type`` to "leafid".







