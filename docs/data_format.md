## **Data Format**

### **Training Data Format**

Ytk-learn supports weighted trainning and sample weight scaling, so each line is a sample with four fields(weight, labels,features and initial prediction), with the last field optional. We provide **data format transform tool** to convert data format from  [LibSVM]( [LibSVM](https://www.csie.ntu.edu.tw/~cjlin/libsvm/)) to ytk-learn:

```weight${x_delim}labels${x_delim}features${x_delim}init_prediction```, the four fields are split by ``${x_delim}``.

+ **Weight** is the sample weight, real num.

- **Labels**  stands for one label or  mutiple labels(```label_1${y_delim}...${y_delim}label_n```). Multiple labels are splitted by ``${y_delim}``.  The label format is related to [training objective](models.md):
  - regression:  real number, e.g. 122.23
  - binary classification: 0 or 1
  - binary cross_entropy: real number, belongs to [0, 1], e.g. 0.245
  - multiclass classficication: one-hot coding,  length of labels is equal to class number, e.g. 0, 0, 1, 0 means that target is class2 in 4 classes(class_0, class_1, class_2, class_3)
  - multiclass cross_entropy: length of labels is equal to class number, sum of labels equals to 1, e.g. 0.2,0.1,0.4,0.3 (4 class in total)


- **Features** stands for ```f1_name${feature_name_val_delim}f1_value${features_delim}...${features_delim}fn_name${feature_name_val_delim}fn_value```. Features are split by ${features_delim} while feature key and value are split by ``${feature_name_val_delim}``.
- **Init_prediction**  is optional, and only is enabled in tree-based models(e.g. GBDT, GBMLR, GBSDT,...). You can provide each sample with an initial prediction(regression and binary classification) or initial scores(multi-class classification, scores are the origin predict score before softmax). When you provide this field, set 'sample_dependent_base_prediction' in model configuration file to true. 

For example, if you config  ``data.delim`` with 

```
x_delim : "###",
y_delim : ",",
features_delim : ",",
feature_name_val_delim : ":"
```

then the data format becomes

```weight###labels###f1_name:f1_value,f2_name:f2_value,...,fn_name:fn_value###init_prediction```, which is similar to LibSVM format. Following are some examples:

- **regression**

  ```
  1###12.45###height:1.6,weight:56.0,size:102
  ```

  The first '1' is the sample weight, '12.45' is the sample label,  'height', 'weight' and 'size'  are feature names,  '1.6', '56.0' and '102' are feature values. If you provide an initial prediction of '0.4' for this sample, then the data sample is as below:

  ```
  1###12.45###height:1.6,weight:56.0,size:102###0.4
  ```


- **binary classification**

   ```
   10###1###height:1.6,weight:56.0,size:102
   1###0###height:2.0,weight:50.0,size:80
   ```

   There are two samples, the first '10' in the first line is the sample weight, the '1' in ``10###1###``is the sample label. Similarly, the second sample has weight '1' and label '0'. In binary classification, label '1' stands for positive sample while '0' stands for negative sample. You can also provide probability values in [0,1] as label, indicating the probability that the sample is positive, e.g.

   ```
   10###0.9###height:1.6,weight:56.0,size:102
   1###0.05###height:2.0,weight:50.0,size:80
   ```
   If you want to provide inital prediction for each sample,  an example is as as below:

   ```
   10###0.9###height:1.6,weight:56.0,size:102###0.8
   1###0.05###height:2.0,weight:50.0,size:80###0.01
   ```

- **multi-class classification**

   ```
   1###0,0,0,1,0,0###height:1.6,weight:56.0,size:102
   ```

   The first '1' is the sample weight, '0,0,0,1,0,0' is the sample label which means target is the fourth class of the six classes. Ytklearn also supports probability values in [0,1] as label, to indicate the probability that the sample belongs to this class, e.g.

   ```
   1###0.01,0.02,0.01,0.75,0.01,0.2###height:1.6,weight:56.0,size:102
   ```

   If you want to provide inital scores for each sample,  an example is as as below:

   ```
   1###0.01,0.02,0.01,0.75,0.01,0.2###height:1.6,weight:56.0,size:102###0.0,0.01,0.01,0.95,0.01,0.02
   ```

   Bear in mind that the sum of label values is 1 in multi-class classification. 

Format of testing data as validation set in training phase is the same as training data, but a little different in  [offline batch evaluation](). 

### Data Transform Script

```bin/transform.py``` python script provides the ability to transfrom data line **in memory** during reading process, so you can do a number of experiments while you don't have to change your original data. 

This script can be used for:

```
1. changing other data format to ytk-learn data format
2. feature transform/scale, features cartesian product, generating polynomial features(x1, x2, x3) -> (x1, x2, x3, x1x2, x1x3, x2x3, x1x2x3)
3. changing sample weight
4. sampling, e.g. negative sampling
5. generating multiple lines
6. using powerful third libs, such as sklearn, pandas, etc., to handle your data
```

Here is a transform script for transforming libsvm data format to ytk-learn data format:

```python
#!/user/bin/env python
# -*- coding: UTF-8 -*-
'''
The main purpose of this python script is to change you data line in reading data process(you don't need to change your original data), transform function can change original line into new lines.

It can be used for:
  1. changing other data formats to ytk-learn data format
  2. feature transform/scale, features cartesian product, generating polynomial features(x1, x2, x3) -> (x1, x2, x3, x1x2, x1x3, x2x3, x1x2x3)
  3. changing sample weight
  4. sampling, e.g. negative sampling
  5. generating multi lines
  6. using powerful third libs, such as sklearn, pandas, etc., to handle your data
'''

# line -> lines, if this line is filtered, return []
def transform(bytesarr):
    # don't delete this code
    line = bytesarr.decode("utf-8")
    info = line.split(" ")
    label = ""
    if info[0] == "-1":
        label =  "0"
    else:
        label = "1"
    features = ",".join(info[1:])
    
    line = "1.0###" + label + "###" + features

    # custom code here
    # ...
    return [line]
```

In default, ```bin/transform.py``` is not used, and you can set ```transform="true"```  to turn it on in the running script.



### **Data Format Transform Tool** ###

Ytklearn provides the script[libsvm_convert_2_ytklearn.sh](../bin/libsvm_convert_2_ytklearn.sh) ([win_libsvm_convert_2_ytklearn.bat](../bin/win_libsvm_convert_2_ytklearn.bat) for windows) to transform data format from libsvm to ytklearn. It will **generate a new dataset file** of ytklearn format and the weight of each sample is one.

| Task                                     | Desc                                     | Example                     | Original Format                          | Converted Format                         |
| ---------------------------------------- | ---------------------------------------- | --------------------------- | ---------------------------------------- | ---------------------------------------- |
| regression                               |                                          | regression                  | 43.2 1:0.3 2:0.9 3:33.2                  | 1###43.2###1:0.3,2:0.9,3:33.2            |
| binary_classification@label1,label2      | label1 is negative, label2 is positive   | binary_classification@-1,1  | 1 1:0.3 2:0.9 3:33.2<br>-1 1:41 2:0.9 3:12 | 1###1###1:0.3 ,2:0.9,3:33.2<br>1###0 ###1:41, 2:0.9,3:12 |
| multi_classification@label1,label2,...,labeln | label1 is class 1, label2 is class 2,...,labeln is class n | binary_classification@1,2,3 | 1 1:0.3 2:0.9 3:33.2<br>2 1:41 2:0.9 3:12<br>3 1:3.2 2:1.1 3:14 | 1###1,0,0###1:0.3,2:0.9, 3:33.2<br>1###0,1,0### 1:41,2:0.9,3:12<br>1###0,0,1### 1:3.2,2:1.1,3:14 |



```bash
#!/usr/bin/env bash
# binary_classification@label1,label2, multi_classification@label1,label2,..., regression
mode="???"
x_delim="###"
y_delim=","
features_delim=","
feature_name_val_delim=":"

# local filesystem : "local", "file:///",
# hdfs filesystem : "hdfs://host"
fs_scheme="???"
libsvm_data_path="???"
ytklearn_data_path="???"

nohup java -server -Xmx1000m -XX:-OmitStackTraceInFastThrow -classpath .:lib/*:config -Dlog4j.configuration=file:config/log4j.properties com.fenbi.ytklearn.utils.LibsvmConvertTool  \
    "${mode}" "${x_delim}" "${y_delim}" "${features_delim}" "${feature_name_val_delim}" "${fs_scheme}" "${libsvm_data_path}" "${ytklearn_data_path}"  >> log/info.log 2>&1 &
```

