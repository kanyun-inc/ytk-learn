#!/usr/bin/env bash
mvn clean
mvn package
mkdir target/ytk-learn
mkdir target/ytk-learn/lib
mkdir target/ytk-learn/log
cp -r bin target/ytk-learn
cp -r config target/ytk-learn
cp -r experiment target/ytk-learn/experiment
cp -r demo target/ytk-learn
cp -r docs target/ytk-learn
cp target/ytk-learn.jar  target/ytk-learn/lib
cd target
zip -r ytk-learn.zip ytk-learn
