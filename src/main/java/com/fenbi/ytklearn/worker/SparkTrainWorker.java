/**
*
* Copyright (c) 2017 ytk-learn https://github.com/yuantiku
*
* Permission is hereby granted, free of charge, to any person obtaining a copy
* of this software and associated documentation files (the "Software"), to deal
* in the Software without restriction, including without limitation the rights
* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
* copies of the Software, and to permit persons to whom the Software is
* furnished to do so, subject to the following conditions:

* The above copyright notice and this permission notice shall be included in all
* copies or substantial portions of the Software.

* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
* SOFTWARE.
*/

package com.fenbi.ytklearn.worker;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function2;

import java.io.Serializable;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;

/**
 * @author xialong
 */

public class SparkTrainWorker extends TrainWorker implements Serializable {
    private int slaveNum;
    public SparkTrainWorker(
                            SparkConf conf,
                            String modelName,
                            String configPath,
                            String configFile,
                            String pyTransformScript,
                            boolean needPyTransform,
                            String loginName,
                            String hostName,
                            int hostPort,
                            int slaveNum,
                            int threadNum) throws Exception {
        super(modelName, configPath, configFile, pyTransformScript, needPyTransform,
                loginName, hostName, hostPort, threadNum);
        this.slaveNum = slaveNum;

        conf.set("spark.files.fetchTimeout", "3200");
        conf.set("spark.network.timeout", "3200");
        conf.set("spark.dynamicAllocation.executorIdleTimeout", "3200");
        conf.set("spark.dynamicAllocation.schedulerBacklogTimeout", "300");
        conf.set("spark.core.connection.auth.wait.timeout", "3200");
        conf.set("spark.memory.fraction", "0.01");
    }

    Function2<Integer, Iterator<String>, Iterator<Boolean>> trainFunc =
            new Function2<Integer, Iterator<String>, Iterator<Boolean>>() {
        @Override
        public Iterator<Boolean> call(Integer v1, Iterator<String> v2) throws Exception {
            LOG.info("###partition:" + v1);
            boolean res = train(Arrays.asList(v2), null);
            List<Boolean> retList = Arrays.asList(res);
            return retList.iterator();
        }
    };

    public boolean sparkTrain(JavaRDD<String> rdd) {
        JavaRDD<String> repartition = rdd.repartition(slaveNum);
        JavaRDD<Boolean> partRDD = repartition.mapPartitionsWithIndex(trainFunc, true);
        List<Boolean> res = partRDD.collect();
        for (boolean result : res) {
            if (!result) {
                return false;
            }
        }
        return true;
    }

    public static void main(String []args) throws Exception {

        String modelName = args[0];
        String configPath = args[1];
        String configFile = args[2];
        String pyTransformScript = args[3];
        boolean needPyTransform = Boolean.parseBoolean(args[4]);
        String loginName = args[5];
        String hostName = args[6];
        int hostPort = Integer.parseInt(args[7]);
        int slaveNum = Integer.parseInt(args[8]);
        int threadNum = Integer.parseInt(args[9]);

        LOG.info("configFile:" + configFile);
        LOG.info("loginName:" + loginName);
        LOG.info("hostName:" + hostName + ", hostPort:" + hostPort);
        LOG.info("slaveNum:" + slaveNum + ", threadNum:" + threadNum);
        LOG.info("modelName:" + modelName);

        SparkConf conf = new SparkConf();
        SparkTrainWorker worker = new SparkTrainWorker(
                conf,
                modelName,
                configPath,
                configFile,
                pyTransformScript,
                needPyTransform,
                loginName,
                hostName,
                hostPort,
                slaveNum,
                threadNum);
        JavaSparkContext sc = new JavaSparkContext(conf);
        String trainDataPath = worker.getTrainDataPath();
        JavaRDD<String> trainRDD = sc.textFile(trainDataPath);
        LOG.info("trainDataPath:" + trainDataPath);

        if (!worker.sparkTrain(trainRDD)) {
            throw new Exception("spark train exception!");
        }

        System.exit(0);
    }
}
