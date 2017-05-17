#!/user/bin/env python
# -*- coding: UTF-8 -*-
'''
This python script main purpose is to change you data line in reading data process(you needn't to change your original data),
transform function change original line to new lines.

can used for:
  1. change other data format to ytk-learn data format
  2. feature transform/scale, features cartesian product, generating polynomial features(x1, x2, x3) -> (x1, x2, x3, x1x2, x1x3, x2x3, x1x2x3)
  3. change sample weight
  4. samples sampling, e.g. negative samples sampling
  5. generate multi lines
  6. you can use sklearn, pandas... powerful third libs to handle your data
'''

# line -> lines, if this line is filtered, return []
def transform(bytesarr):
    # don't delete this code
    line = bytesarr.decode("utf-8")
    # custom code here
    # ...
    cols = line.split(' ')
    label = cols[0]
    feas = ','.join(cols[1:])
    new_line = '###'.join(['1', label, feas])
    return [new_line]