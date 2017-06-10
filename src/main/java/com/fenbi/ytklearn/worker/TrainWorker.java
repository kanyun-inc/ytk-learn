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

import com.fenbi.ytklearn.dataflow.*;
import com.fenbi.ytklearn.fs.FileSystemFactory;
import com.fenbi.ytklearn.fs.IFileSystem;
import com.fenbi.ytklearn.optimizer.OptimizerFactory;
import com.fenbi.ytklearn.operation.ITrainOperation;
import com.fenbi.ytklearn.operation.TrainOperationFactory;
import com.fenbi.mp4j.exception.Mp4jException;
import com.fenbi.mp4j.comm.ThreadCommSlave;
import com.fenbi.ytklearn.utils.CheckUtils;
import com.typesafe.config.Config;
import com.typesafe.config.ConfigFactory;
import com.typesafe.config.ConfigValue;
import com.typesafe.config.ConfigValueFactory;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.Serializable;
import java.net.URI;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

/**
 * @author xialong
 */

public class TrainWorker implements Serializable {
    public static final Logger LOG = LoggerFactory.getLogger(TrainWorker.class);

    protected String modelName;
    protected String configPath;
    protected String configFile;
    protected String pyTransformScript;
    protected boolean needPyTransform;
    protected String loginName;
    protected String hostName;
    protected int hostPort;
    protected int threadNum;

    final Map<String, Object> customParamsMap = new HashMap<>();

    public TrainWorker( String modelName,
                        String configPath,
                        String configFile,
                        String pyTransformScript,
                        boolean needPyTransform,
                        String loginName,
                        String hostName,
                        int hostPort,
                        int threadNum
                             ) throws Exception {

        this.modelName = modelName;
        this.configPath = configPath;
        this.configFile = configFile;
        this.pyTransformScript = pyTransformScript;
        this.needPyTransform = needPyTransform;
        this.loginName = loginName;
        this.hostName = hostName;
        this.hostPort = hostPort;
        this.threadNum = threadNum;

        LOG.info("configFile:" + configFile);
        LOG.info("configPath:" + configPath);
        LOG.info("pyTransformScript:" + pyTransformScript);
        LOG.info("loginName:" + loginName);
        LOG.info("hostName:" + hostName + ", hostPort:" + hostPort);
        LOG.info("threadNum:" + threadNum);
        LOG.info("modelName:" + modelName);
    }


    public String getTrainDataPath() {
        String configRealPath = (new File(configFile).exists()) ? configFile : configPath;
        File realFile = new File(configRealPath);
        CheckUtils.check(realFile.exists(), "config file(%s) doesn't exist!", configRealPath);
        Config config = ConfigFactory.parseFile(realFile);
        config = updateConfigWithCustom(config);
        return config.getString("data.train.data_path");
    }

    public String getURI() {
        String configRealPath = (new File(configFile).exists()) ? configFile : configPath;
        File realFile = new File(configRealPath);
        CheckUtils.check(realFile.exists(), "config file(%s) doesn't exist!", configRealPath);
        Config config = ConfigFactory.parseFile(realFile);
        config = updateConfigWithCustom(config);
        return config.getString("fs_scheme");
    }

    private Config updateConfigWithCustom(Config config) {
        for (Map.Entry<String, Object> entry : customParamsMap.entrySet()) {
            config = config.withValue(entry.getKey(), ConfigValueFactory.fromAnyRef(entry.getValue()));
        }
        return config;
    }

    public void emptyCustomParams() {
        customParamsMap.clear();
    }

    public void setCustomParam(String key, Object value) {
        customParamsMap.put(key, value);
    }

    public boolean train(List<Iterator<String>> trainDatas,
                      List<Iterator<String>> testDatas) {
        long start = System.currentTimeMillis();
        int errorCode = 0;
        ThreadCommSlave comm = null;
        try {
            comm = new ThreadCommSlave(loginName, threadNum, hostName, hostPort);
            File file = new File(configFile);
            CheckUtils.check(file.exists(), "config file(%s) doesn't exist!", configFile);
            Config config = ConfigFactory.parseFile(file);
            config = updateConfigWithCustom(config);

            comm.info("################ parameters ################");
            for (Map.Entry<String, ConfigValue> entry : config.entrySet()) {
                comm.info(entry.getKey() + "=" + entry.getValue());
            }

            String uri = config.getString("fs_scheme");
            LOG.info("file system uri:" + uri + ", URI:" + new URI(uri) + ", URI tostring:" + (new URI(uri)).toString());

            IFileSystem fs = FileSystemFactory.createFileSystem(new URI(uri));

            DataFlow dataFlow = DataFlowFactory.createDataFlow(modelName, fs, config,
                    comm, threadNum, needPyTransform, pyTransformScript);
            dataFlow.init();
            dataFlow.loadFlow(trainDatas, testDatas);
            dataFlow.ready();
            long beforeTrainCost = System.currentTimeMillis() - start;

            ITrainOperation trainOperation = TrainOperationFactory.createTrainOperation(modelName);

            final ThreadCommSlave finalComm = comm;

            // 开始训练
            Thread []threads = new Thread[threadNum];
            for (int t = 0; t < threadNum; t++) {
                final int tidx = t;
                threads[t] = new Thread(t + "") {
                    @Override
                    public void run() {
                        finalComm.setThreadId(tidx);
                        try {
                            if (!dataFlow.isReady()) {
                                throw new Exception("data flow is not ready!");
                            }

                            trainOperation.operate(dataFlow,
                                    OptimizerFactory.createOptimizer(
                                            modelName,
                                            dataFlow,
                                            tidx),
                                    finalComm,
                                    tidx
                            );

                        } catch (Exception e) {
                            try {
                                finalComm.exception(e);
                                finalComm.close(1);
                            } catch (Mp4jException e1) {
                                LOG.error("comm send exception message error!", e);
                            }
                            System.exit(1);
                        }
                    }
                };
                threads[t].start();
            }

            for (int t = 0; t < threadNum; t++) {
                threads[t].join();
            }

            long totalCost = System.currentTimeMillis() - start;
            long trainCost =  totalCost - beforeTrainCost;

            comm.info(String.format("\nTrain cost details:\n%-19s%9.2fs\n%-19s%9.2fs\n%-19s%9.2fs\n",
                        "LoadDataFlow:", beforeTrainCost / 1000.,
                        "PreprocessAndTrain:", trainCost / 1000.,
                        "Total:", totalCost / 1000.));

        } catch (Exception e) {
            errorCode = 1;
            LOG.error("existed exception!", e);
            if (comm != null) {
                try {
                    comm.exception(e);
                } catch (Mp4jException e1) {
                    LOG.error("comm send exception message error!", e);
                }
            }
        } finally {
            try {
                if (comm != null) {
                    comm.close(errorCode);
                }
            } catch (Mp4jException e) {
                errorCode = 1;
                LOG.error("comm close exception!", e);
            }
        }

        return errorCode == 0;
    }


}
