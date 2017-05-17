import sys

def transform(in_name, out_name):
    fin = open(in_name, 'r')
    fout = open(out_name, 'w')
    for line in fin:
        cols = line.split(' ')
        label = cols[0]
        feas = ','.join(cols[1:])
        new_line = '###'.join(['1', label, feas])
        fout.write(new_line)
    fin.close() 
    fout.close()

f1 = 'higgs.train'
f2 = 'higgs.test'

transform(f1, 'higgs.train.ytklearn')
transform(f2, 'higgs.test.ytklearn')

