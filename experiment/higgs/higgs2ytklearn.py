import os

input_filename = "HIGGS.csv"
output_train = "higgs.train"
output_test = "higgs.test"

num_train = 10500000

read_num = 0

input = open(input_filename, "r")
train = open(output_train, "w")
test = open(output_test,"w")

def WriteOneLine(tokens, output):
    label = int(float(tokens[0]))
    head = '1###%d###' % label 
    new_list = []
    for i in xrange(1,len(tokens)):
        feature_value = float(tokens[i])
        new_list.append('%d:%s' % (i-1, str(feature_value)))
    new_line = head + ','.join(new_list) + '\n'
    output.write(new_line)

line = input.readline()
while line:
    tokens = line.split(',')
    if read_num < num_train:
        WriteOneLine(tokens, train)
    else:
        WriteOneLine(tokens, test)
    read_num += 1
    line = input.readline()

input.close()
train.close()
test.close()
