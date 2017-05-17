#!/bin/bash

echo "Attention: make sure you are in current path (ytk-learn/demo/higgs)"

# download data and put in current path(ytk-learn/demo/higgs)
function download_data() {
    if [ ! -f "HIGGS.csv" ]; then
        if [ ! -f "HIGGS.csv.gz" ]; then
            echo "start to download data HIGGS.csv.gz"
            wget https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz
            echo "download data complete!"
        fi
        gunzip HIGGS.csv.gz
    else
        echo "HIGGS.csv already exists"
    fi
}

# generate train and test data
if [ ! -f "higgs.train" ] || [ ! -f "higgs.test" ]; then
    download_data
    python higgs2ytklearn.py 
    echo "data process complete!"
else
    echo "higgs.train and higgs.test already exists"
fi
