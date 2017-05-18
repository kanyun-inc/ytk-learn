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

package com.fenbi.ytklearn.dataflow;

import com.fenbi.mp4j.comm.ThreadCommSlave;
import com.fenbi.mp4j.exception.Mp4jException;
import com.fenbi.ytklearn.exception.YtkLearnException;
import com.fenbi.ytklearn.fs.IFileSystem;
import com.fenbi.ytklearn.loss.ILossFunction;
import com.fenbi.ytklearn.param.DataParams;
import com.fenbi.ytklearn.param.FeatureParams;
import com.fenbi.ytklearn.utils.CheckUtils;
import com.fenbi.ytklearn.utils.LogUtils;
import com.typesafe.config.Config;
import lombok.Data;
import org.python.core.PyFunction;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.*;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.LinkedBlockingDeque;

/**
 * @author xialong
 */

@Data
public abstract class DataFlow {
    public static final Logger LOG = LoggerFactory.getLogger(DataFlow.class);
    public static final String FEATURE_TRANSFORM_STAT_PATH_SUFFIX = "_feature_transform_stat";

    @Data
    public static class CoreParams {
        public String x_delim;
        public String y_delim;
        public String features_delim;
        public String feature_name_val_delim;

        public boolean needYSampling;
        public float[] ySampling;
        public boolean assigned;
        public DataParams.UnassignedMode unassigned_mode;
        public String train_data_path;
        public int train_max_error_tol;
        public String test_data_path;
        //public String test_data_temp_path;
        public int test_max_error_tol;

        public boolean need_dict;
        public String dict_path;

        public boolean need_bias;
        public String bias_feature_name;
        public boolean justEvaluation;
        public String modelPath;

        public ILossFunction lossFunction;

        public boolean needYStat;

        public boolean need_feature_hash;
        public boolean verbose;
        public FeatureParams featureParams = null;
        public boolean continue_train;
    }
    protected Config config;
    protected CoreParams coreParams;

    protected boolean loadingTrainData = true;
    protected String loadingPrefix = "[train data]";

    protected CoreData[] threadTrainCoreDatas;
    protected CoreData[] threadTestCoreDatas;


    protected boolean needPyTransform;
    protected PyFunction pyTransformFunc;

    protected ThreadCommSlave comm;
    protected int rank;
    protected int slaveNum;

    protected LogUtils LOG_UTILS;

    protected int threadNum;

    protected Map<String, Integer> fName2IndexMap;
    protected Map<Integer, String> fIndex2NameMap;

    protected IFileSystem fs;

    protected long gTrainRealSampleNum;
    protected double gTrainWeightSampleNum;
    protected long gTestRealSampleNum;
    protected double gTestWeightSampleNum;
    protected boolean needTest = false;

    protected int dim;

    private volatile boolean ready = false;

    protected volatile boolean replacedIdx = false;
    protected Random rand = new Random();

    private static final BlockingQueue<String> readQueues[];
    private static volatile boolean readFinished = false;

    protected IFeatureMap featureMap;

    protected boolean needFeatureTransform;
    protected Map<Integer, CoreData.TransformNode> transformNodeMap = new HashMap<>();
    protected volatile boolean trainReplacedFeatureTransform = false;
    protected volatile boolean testReplacedFeatureTransform = false;

    public static int MAX_THREAD_NUM = 2000;

    static {
        readQueues = new LinkedBlockingDeque[MAX_THREAD_NUM];
        for (int t = 0; t < MAX_THREAD_NUM; t++) {
            readQueues[t] = new LinkedBlockingDeque<>();
        }
    }

    public static class ThreadIterator implements Iterator<String> {
        private final int threadId;

        public ThreadIterator(int threadId) {
            this.threadId = threadId;
        }

        @Override
        public boolean hasNext() {
            while(!readFinished) {
                if (readQueues[threadId].size() > 0) {
                    return true;
                }
            }

            return readQueues[threadId].size() > 0;
        }

        @Override
        public String next() {
            try {
                String val = readQueues[threadId].take();
                return val;
            } catch (InterruptedException e) {
                e.printStackTrace();
                System.exit(1);
            }
            return null;
        }
    }

    public DataFlow(IFileSystem fs,
                    Config config,
                    ThreadCommSlave comm,
                    int threadNum,
                    boolean needPyTransform,
                    String pyTransformScript
                    ) throws IOException {
        this.config = config;
        this.fs = fs;
        this.comm = comm;
        this.rank = comm.getRank();
        this.slaveNum = comm.getSlaveNum();
        this.threadNum = threadNum;
        this.LOG_UTILS = new LogUtils(comm, config.getBoolean("verbose"));
        this.needPyTransform = needPyTransform;
        this.pyTransformFunc = DataUtils.getTranformFunction(needPyTransform, pyTransformScript);

        this.threadTrainCoreDatas = new CoreData[threadNum];
        this.threadTestCoreDatas = new CoreData[threadNum];

        if (threadNum > MAX_THREAD_NUM) {
            throw new YtkLearnException("thread number=" + threadNum + " is too large! please reset!");
        }
    }

    public void init() throws Exception {
        this.coreParams = createCoreParams();

        needTest = coreParams.test_data_path != null && coreParams.test_data_path.trim().length() > 0;
        needFeatureTransform = coreParams.featureParams != null &&
                coreParams.featureParams.transform != null &&
                coreParams.featureParams.transform.switch_on;

        LOG_UTILS.verboseInfo(getParams());
    }

    public String getParams() {
        return "DataFlow{" +
                "coreParams=" + coreParams +
                ", rank=" + rank +
                ", slaveNum=" + slaveNum +
                ", threadNum=" + threadNum +
                '}';
    }

    public void ready() {
        ready = true;
    }

    public boolean isReady() {
        return ready;
    }

    public abstract CoreParams createCoreParams() throws Exception;

    protected abstract CoreData getCoreData();

    protected int getYnum() {
        return 1;
    }

    protected abstract boolean aheadLoadModel();


    protected void loadDict() throws IOException, Mp4jException {
        String modelDictPath = coreParams.modelPath + "_dict";
        String dictPathDir = coreParams.justEvaluation ? modelDictPath : coreParams.dict_path;

        if (coreParams.justEvaluation) {
            coreParams.need_dict = true;
            LOG_UTILS.importantInfo("just evaluation, so load this model's dict, path:" + dictPathDir);
        } else {
            if (!coreParams.need_dict && coreParams.continue_train && fs.exists(modelDictPath)) {
                coreParams.need_dict = true;
                dictPathDir = modelDictPath;
                LOG_UTILS.importantInfo("continue_train=true && model dict path exist=" + modelDictPath + ", will be loaded!");
            }
        }

        if (!coreParams.need_dict) {
            LOG_UTILS.importantInfo("have no dict, we will collect feature dict...");
            return;
        }

        if (fName2IndexMap == null) {
            fName2IndexMap = new HashMap<>();
        }

        if (coreParams.need_bias) {
            fName2IndexMap.put(coreParams.bias_feature_name, 0);
        }

        List<String> dictPaths = fs.recurGetPaths(Arrays.asList(dictPathDir));
        Collections.sort(dictPaths);
        for (int i = 0; i < dictPaths.size(); i++) {
            String dictPath = dictPaths.get(i);
            BufferedReader reader = new BufferedReader(fs.getReader(dictPath));
            String line;
            while ((line = reader.readLine()) != null) {
                fName2IndexMap.put(line.trim(), fName2IndexMap.size());
            }
        }

        LOG_UTILS.importantInfo("load dict finished! dict size:" + fName2IndexMap.size());

    }


    protected void setDim() throws Mp4jException {
        dim = fName2IndexMap.size();
        CheckUtils.check(dim > 0, "feature dim(%d) <= 0 is invalid! may be cased by no data or filter all feature", dim);
        LOG_UTILS.importantInfo("dim:" + dim);
    }

    protected void reduceFeature() throws Mp4jException, IOException {

        if (!coreParams.need_dict) {
            Map<String, Long> featureCntMap = threadTrainCoreDatas[0].getFeatureCntMap();
            LOG_UTILS.importantInfo("before feature filtering, feature size:" + featureCntMap.size());
            TreeSet<String> nameSet = new TreeSet<>();
            for (Map.Entry<String, Long> entry : featureCntMap.entrySet()) {
                if (entry.getValue() >= coreParams.featureParams.filter_threshold) {
                    nameSet.add(entry.getKey());
                }
            }

            LOG_UTILS.importantInfo("global feature number:" + nameSet.size() +
                    ", filtered feature number:" + (featureCntMap.size() - nameSet.size()));

            if (coreParams.need_bias) {
                nameSet.remove(coreParams.bias_feature_name);
            }

            int fsize = nameSet.size();
            for (int t = 0; t < threadNum; t++) {
                threadTrainCoreDatas[t].releaseFeatureCntMap();
            }

            LOG_UTILS.verboseInfo("begin gen feature name to index map...");
            fName2IndexMap = new HashMap<>(fsize + 1);
            if (coreParams.need_bias) {
                fName2IndexMap.put(coreParams.bias_feature_name, 0);
            }

            int delta = coreParams.need_bias ? 1 : 0;
            int i = 0;
            for (String fn : nameSet) {
                fName2IndexMap.put(fn, i + delta);
                i++;
            }
            LOG_UTILS.importantInfo("feature name to index map! size:" + fName2IndexMap.size());
        }

        if (needFeatureTransform) {
            Set<String> transformFeatureSet = new HashSet<>();
            Set<String> includeSet = coreParams.featureParams.transform.include;
            Set<String> excludeSet = coreParams.featureParams.transform.exclude;
            Set<String> nameSet = new HashSet<>();
            for (String name : fName2IndexMap.keySet()) {
                nameSet.add(name);
            }
            if (coreParams.need_bias) {
                nameSet.remove(coreParams.bias_feature_name);
            }
            if (includeSet != null && includeSet.size() > 0) {
                transformFeatureSet = includeSet;
            } else if (excludeSet != null && excludeSet.size() > 0){
                transformFeatureSet.addAll(nameSet);
                for (String fn : excludeSet) {
                    transformFeatureSet.remove(fn);
                }
            } else {
                transformFeatureSet.addAll(nameSet);
            }

            LOG_UTILS.verboseInfo("need transform feature size:" + transformFeatureSet.size() + "," + transformFeatureSet);

            String transformStatPath = coreParams.modelPath + FEATURE_TRANSFORM_STAT_PATH_SUFFIX;
            LOG_UTILS.importantInfo("feature transform stat save path:" + transformStatPath);
            PrintWriter statWriter = new PrintWriter(fs.getWriter(transformStatPath));
            Map<String, CoreData.FeatureStat> featureStat = threadTrainCoreDatas[0].getFeatureStat();
            for (String fn : transformFeatureSet) {
                if (!featureStat.containsKey(fn)) {
                    LOG_UTILS.error("transform feature:" + fn + " may be not existing!");
                    continue;
                }
                CoreData.FeatureStat fstat = featureStat.get(fn);
                CoreData.TransformNode transformNode = fstat.convert(coreParams.featureParams.transform.mode,
                        coreParams.featureParams.transform.scale_range.max,
                        coreParams.featureParams.transform.scale_range.min
                );
                transformNodeMap.put(fName2IndexMap.get(fn), transformNode);
                statWriter.println(fn + "###" + transformNode.toString());
            }
            statWriter.close();

            LOG_UTILS.verboseInfo("transform feature map:" + transformNodeMap);
        }
    }




    protected abstract void loadModel() throws IOException, Mp4jException, ClassNotFoundException, InterruptedException;

    protected abstract void handleOtherTrainInfo() throws Mp4jException;

    protected abstract void handleOtherTestInfo() throws Mp4jException;

    public abstract void dumpModel() throws IOException, Mp4jException;

    protected List<Iterator<String>> getAssignedDatas(String dataPath) throws Exception {
        List<Iterator<String>> datas;
        if (coreParams.assigned) {
            datas = fs.read(Arrays.asList(dataPath));
        } else {
            if (coreParams.unassigned_mode == DataParams.UnassignedMode.FILES_AVG) {
                List<String> paths = fs.recurGetPaths(Arrays.asList(dataPath));
                Collections.sort(paths);

                int [][]avgAssign = DataUtils.avgAssign(paths.size(), slaveNum);
                int start = avgAssign[rank][0];
                int end = avgAssign[rank][1];
                datas = fs.read(paths.subList(start, end));
            } else {
                datas = fs.selectRead(Arrays.asList(dataPath), slaveNum, rank);
            }

        }
        return datas;
    }


    public void handleLocalIdx(int tidx, Map<Integer, String> fIndex2NameMap) {
        if (replacedIdx) {
            return;
        }

        CoreData coreData = threadTrainCoreDatas[tidx];
        int x[][] = coreData.x;
        int[] realNum = coreData.realNum;
        int xidx[][] = coreData.xidx;

        for (int k = 0; k < coreData.cursor2d; k++) {
            int lsNumInt = realNum[k];
            int suballMissCnt = 0;
            for (int i = 0; i < lsNumInt; i++) {
                int start = xidx[k][i];
                int end = xidx[k][i + 1];

                xidx[k][i] -= suballMissCnt;
                for (int j = start; j < end; j += 2) {
                    int localIdx = x[k][j];
                    String fname = fIndex2NameMap.get(localIdx);
                    Integer gidx = fName2IndexMap.get(fname);
                    if (gidx != null) {
                        x[k][j] = gidx;
                        x[k][j - suballMissCnt] = x[k][j];
                        x[k][j + 1 - suballMissCnt] = x[k][j + 1];
                    } else {
                        suballMissCnt += 2;
                    }
                }
            }
            xidx[k][lsNumInt] -= suballMissCnt;
        }
    }

    public void replaceFeatureTransform(CoreData coreData) {

        int x[][] = coreData.x;
        int[] realNum = coreData.realNum;
        int xidx[][] = coreData.xidx;
        for (int k = 0; k < coreData.cursor2d; k++) {
            int lsNumInt = realNum[k];
            for (int i = 0; i < lsNumInt; i++) {
                for (int j = xidx[k][i]; j < xidx[k][i + 1]; j += 2) {
                    if (x[k][j] == 0 && coreParams.need_bias) {
                        continue;
                    }
                    CoreData.TransformNode node = transformNodeMap.get(x[k][j]);
                    x[k][j + 1] = Float.floatToRawIntBits(node.transform(Float.intBitsToFloat(x[k][j + 1])));
                }
            }
        }

    }

    public void loadFlow(final List<Iterator<String>> trainData,
                         final List<Iterator<String>> testData) throws Exception {

        long start = System.currentTimeMillis();
        if (aheadLoadModel()) {
            loadModel();
        }

        // load dict
        loadDict();

        LOG_UTILS.importantInfo("#########read train data############");
        List<Iterator<String>> trainDataNow = trainData != null ? trainData :
                getAssignedDatas(coreParams.train_data_path);
        // just read train data thread
        Thread trainReadThread = new Thread() {
            @Override
            public void run() {
                Random rand = new Random();
                try {
                    for (Iterator<String> iter : trainDataNow) {
                        while(iter.hasNext()) {
                            int tidx = rand.nextInt(threadNum);
                            readQueues[tidx].put(iter.next());
                        }
                    }
                } catch (Exception e) {
                    try {
                        LOG_UTILS.exception(e);
                    } catch (Mp4jException e1) {
                        e1.printStackTrace();
                    }
                    System.exit(1);
                }

            }
        };
        trainReadThread.start();

        // handle train data thread
        featureMap = coreParams.need_dict ? new SimpleFeatureMap(fName2IndexMap) : new OnlineFeatureMap(threadNum);
        Thread []trainCoreDataThreads = new Thread[threadNum];
        for (int t = 0; t < threadNum; t++) {
            final int tidx = t;
            trainCoreDataThreads[t] = new Thread() {
                @Override
                public void run() {
                    try {
                        comm.setThreadId(tidx);
                        threadTrainCoreDatas[tidx] = getCoreData(); // TODO: set featureMap
                        threadTrainCoreDatas[tidx].initAssistData();
                        threadTrainCoreDatas[tidx].readData(new ThreadIterator(tidx), true, getYnum());
                        threadTrainCoreDatas[tidx].globalSync();
                    } catch (Exception e) {
                        try {
                            LOG_UTILS.exception(e);
                        } catch (Mp4jException e1) {
                            e1.printStackTrace();
                        }
                        System.exit(1);
                    }


                }
            };
            trainCoreDataThreads[t].start();
        }

        trainReadThread.join();
        readFinished = true;
        for (int t = 0; t < threadNum; t++) {
            trainCoreDataThreads[t].join();
        }

        // error num
        LOG_UTILS.importantInfo(loadingPrefix + " global data format error number:" + threadTrainCoreDatas[0].getGErrorNum());

        // sample num
        LOG_UTILS.importantInfo(loadingPrefix + " global sample number:" + threadTrainCoreDatas[0].getGRealNum() +
                ", global sample weight sum:" + threadTrainCoreDatas[0].getGWeightNum());
        gTrainRealSampleNum = threadTrainCoreDatas[0].getGRealNum();
        gTrainWeightSampleNum = threadTrainCoreDatas[0].getGWeightNum();

        if (coreParams.needYStat) {
            LOG_UTILS.importantInfo(loadingPrefix + " global sample number for each label:" + Arrays.toString(threadTrainCoreDatas[0].getGYRealNumStat()));
            LOG_UTILS.importantInfo(loadingPrefix + " global sample weight sum for each label:" + Arrays.toString(threadTrainCoreDatas[0].getGYWeightNumStat()));
        }

        // reduce feature
        reduceFeature();

        // set dim
        setDim();

        if (!coreParams.need_dict) {
            fIndex2NameMap = ((OnlineFeatureMap) featureMap).getIndex2NameMap(coreParams.need_bias, coreParams.bias_feature_name);
            ((OnlineFeatureMap) featureMap).release();
            // replace idx
            Thread []replaceThreads = new Thread[threadNum];
            for (int t = 0; t < threadNum; t++) {
                final int tidx = t;
                replaceThreads[t] = new Thread() {
                    @Override
                    public void run() {
                        try {
                            handleLocalIdx(tidx, fIndex2NameMap);
                        } catch (Exception e) {
                            try {
                                LOG_UTILS.exception(e);
                            } catch (Mp4jException e1) {
                                e1.printStackTrace();
                            }
                            System.exit(1);
                        }


                    }
                };
                replaceThreads[t].start();
            }

            for (int t = 0; t < threadNum; t++) {
                replaceThreads[t].join();
            }
            replacedIdx = true;
            fIndex2NameMap.clear();


        }

        // train feature transform
        if (needFeatureTransform && !trainReplacedFeatureTransform) {
            Thread []replaceThreads = new Thread[threadNum];
            for (int t = 0; t < threadNum; t++) {
                final int tidx = t;
                replaceThreads[t] = new Thread() {
                    @Override
                    public void run() {
                        try {
                            replaceFeatureTransform(threadTrainCoreDatas[tidx]);
                        } catch (Exception e) {
                            try {
                                LOG_UTILS.exception(e);
                            } catch (Mp4jException e1) {
                                e1.printStackTrace();
                            }
                            System.exit(1);
                        }


                    }
                };
                replaceThreads[t].start();
            }

            for (int t = 0; t < threadNum; t++) {
                replaceThreads[t].join();
            }
            trainReplacedFeatureTransform = true;
        }

        // reduce other train info
        handleOtherTrainInfo();


        if (needTest) {

            loadingTrainData = false;
            loadingPrefix = "[test data]";
            readFinished = false;

            // read test data
            LOG_UTILS.importantInfo("#########read test data############");
            List<Iterator<String>> testDataNow = testData != null ? testData :
                    getAssignedDatas(coreParams.test_data_path);

            // just read test data thread
            Thread testReadThread = new Thread() {
                @Override
                public void run() {
                    Random rand = new Random();
                    try {
                        for (Iterator<String> iter : testDataNow) {
                            while(iter.hasNext()) {
                                int tidx = rand.nextInt(threadNum);
                                readQueues[tidx].put(iter.next());
                            }
                        }
                    } catch (Exception e) {
                        try {
                            LOG_UTILS.exception(e);
                        } catch (Mp4jException e1) {
                            e1.printStackTrace();
                        }
                        System.exit(1);
                    }


                }
            };
            testReadThread.start();

            // handle test data thread
            featureMap = new SimpleFeatureMap(fName2IndexMap);
            Thread []testCoreDataThreads = new Thread[threadNum];
            for (int t = 0; t < threadNum; t++) {
                final int tidx = t;
                testCoreDataThreads[t] = new Thread() {
                    @Override
                    public void run() {
                        try {
                            comm.setThreadId(tidx);
                            threadTestCoreDatas[tidx] = getCoreData(); // TODO: set featureMap
                            threadTestCoreDatas[tidx].initAssistData();
                            threadTestCoreDatas[tidx].readData(new ThreadIterator(tidx), false, getYnum());
                            threadTestCoreDatas[tidx].globalSync();
                        } catch (Exception e) {
                            try {
                                LOG_UTILS.exception(e);
                            } catch (Mp4jException e1) {
                                e1.printStackTrace();
                            }
                            System.exit(1);
                        }


                    }
                };
                testCoreDataThreads[t].start();
            }
            testReadThread.join();
            readFinished = true;
            for (int t = 0; t < threadNum; t++) {
                testCoreDataThreads[t].join();
            }

            // error num
            LOG_UTILS.importantInfo(loadingPrefix + " global data format error number:" + threadTestCoreDatas[0].getGErrorNum());

            // sample num
            LOG_UTILS.importantInfo(loadingPrefix + " global sample number:" + threadTestCoreDatas[0].getGRealNum() +
                    ", global sample weight sum:" + threadTestCoreDatas[0].getGWeightNum());
            gTestRealSampleNum = threadTestCoreDatas[0].getGRealNum();
            gTestWeightSampleNum = threadTestCoreDatas[0].getGWeightNum();

            if (coreParams.needYStat) {
                LOG_UTILS.importantInfo(loadingPrefix + " global sample number for each label:" + Arrays.toString(threadTestCoreDatas[0].getGYRealNumStat()));
                LOG_UTILS.importantInfo(loadingPrefix + " global sample weight sum for each label:" + Arrays.toString(threadTestCoreDatas[0].getGYWeightNumStat()));
            }

            // test feature transform
            if (needFeatureTransform && !testReplacedFeatureTransform) {
                Thread []replaceThreads = new Thread[threadNum];
                for (int t = 0; t < threadNum; t++) {
                    final int tidx = t;
                    replaceThreads[t] = new Thread() {
                        @Override
                        public void run() {
                            try {
                                replaceFeatureTransform(threadTestCoreDatas[tidx]);
                            } catch (Exception e) {
                                try {
                                    LOG_UTILS.exception(e);
                                } catch (Mp4jException e1) {
                                    e1.printStackTrace();
                                }
                                System.exit(1);
                            }


                        }
                    };
                    replaceThreads[t].start();
                }

                for (int t = 0; t < threadNum; t++) {
                    replaceThreads[t].join();
                }
                testReplacedFeatureTransform = true;
            }

            for (int t = 0; t < threadNum; t++) {
               transformNodeMap = null;
            }


            // reduce other test info
            handleOtherTestInfo();
        }

        if (!aheadLoadModel()) {
            // load model
            loadModel();
        }

        LOG_UTILS.importantInfo("load data takes = " + ((System.currentTimeMillis() - start) / 1000.) + "s");
    }
}
