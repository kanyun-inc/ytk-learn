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
	output.write(str(label))
	for i in xrange(1,len(tokens)):
		feature_value = float(tokens[i])
		output.write(' ' + str(i-1) + ':' + str(feature_value))
	output.write('\n')

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
