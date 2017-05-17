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

import com.fenbi.ytklearn.fs.FileSystemFactory;
import com.fenbi.ytklearn.fs.IFileSystem;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.GenericOptionsParser;

import java.io.*;
import java.net.URI;
import java.net.URLDecoder;
import java.net.URLEncoder;
import java.util.*;
import java.util.concurrent.LinkedBlockingQueue;

/**
 * @author xialong
 */

public class HadoopTrainWorker extends TrainWorker {
    public static final Logger LOG = LoggerFactory.getLogger(HadoopTrainWorker.class);
    String hadoopQueueName;
    String hadoopReduceMemory;
    int slaveNum;
    Configuration conf;
    String user;

    public HadoopTrainWorker(Configuration conf,
                             String modelName,
                             String configPath,
                             String configFile,
                             String pyTransformScript,
                             boolean needPyTransform,
                             String loginName,
                             String hostName,
                             int hostPort,
                             int slaveNum,
                             int threadNum,
                             String hadoopQueueName,
                             String hadoopReduceMemory,
                             String user) throws Exception {
        super(modelName, configPath, configFile, pyTransformScript, needPyTransform, loginName, hostName, hostPort, threadNum);
        this.conf = conf;
        this.slaveNum = slaveNum;
        this.hadoopQueueName = hadoopQueueName;
        this.hadoopReduceMemory = hadoopReduceMemory;
        this.user = user;

        conf.set("mapreduce.task.timeout", "720000000");
        conf.set("modelName", modelName);
        conf.set("configFile", configFile);
        conf.set("pyTransformScript", pyTransformScript);
        conf.setBoolean("needPyTransform", needPyTransform);
        conf.set("loginName", loginName);
        conf.set("hostName", hostName);
        conf.setInt("hostPort", hostPort);
        conf.setInt("slaveNum", slaveNum);
        conf.setInt("threadNum", threadNum);

        conf.set("mapreduce.job.queuename", hadoopQueueName);
        conf.set("mapreduce.reduce.memory.mb", hadoopReduceMemory);
        conf.set("mapreduce.reduce.java.opts", "-Xmx" + ((int)((Integer.parseInt(hadoopReduceMemory) * 0.9))) + "m");
        conf.set("yarn.app.mapreduce.am.resource.mb", "" + threadNum);
        conf.set("mapreduce.reduce.cpu.vcores", "" + threadNum);
    }

    public static String encodeMap(Map<String, Object> map) throws Exception {
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        ObjectOutputStream oos = new ObjectOutputStream(baos);
        oos.writeObject(map);
        oos.flush();
        oos.close();
        return URLEncoder.encode(baos.toString("ISO-8859-1"), "UTF-8");
    }

    public static Map<String, Object> decodeMap(String info) throws Exception {
        info = URLDecoder.decode(info, "UTF-8");
        ObjectInputStream ois = new ObjectInputStream(new ByteArrayInputStream(info.getBytes("ISO-8859-1")));
        Map<String, Object> map = (Map<String, Object>)ois.readObject();
        ois.close();
        return map;
    }

    public boolean hadoopTrain() {
//        LOG.info("getclass" + GenericsUtil.getClass(customParamsMap));
//        DefaultStringifier<Map<String, Object>> mapStringifier =
//                new DefaultStringifier<>(conf, GenericsUtil.getClass(customParamsMap));
//        String customParamsMapStr = mapStringifier.toString(customParamsMap);
//        conf.set("customParamsMap", customParamsMapStr);

        boolean sucess;
        String outputPath = null;
        try {
            conf.set("customParamsMap", encodeMap(customParamsMap));

            Job job = Job.getInstance(conf, user + " " + modelName + " training on hadoop");
            String trainDataPath = getTrainDataPath();

            job.setJarByClass(HadoopTrainWorker.class);
            job.setMapperClass(TrainMapper.class);
            job.setOutputKeyClass(Text.class);
            job.setOutputValueClass(Text.class);
            job.setReducerClass(TrainReducer.class);
            FileInputFormat.addInputPath(job, new Path(trainDataPath));
            outputPath = trainDataPath + "_temp_will_be_deleted";
            IFileSystem fs = FileSystemFactory.createFileSystem(new URI(getURI()));
            if (outputPath != null) {
                fs.delete(outputPath);
            }
            FileOutputFormat.setOutputPath(job, new Path(outputPath));
            job.setNumReduceTasks(slaveNum);


            sucess = job.waitForCompletion(true);
        } catch (Exception e) {
            sucess = false;
            LOG.error("hadoop train exception!", e);
        } finally {
            try {
                IFileSystem fs = FileSystemFactory.createFileSystem(new URI(getURI()));
                if (outputPath != null) {
                    fs.delete(outputPath);
                }
            } catch (Exception e) {
                sucess = false;
                LOG.error("hadoop train exception!", e);
            }
        }

        return sucess;
    }

    public static class HadoopTrainWorkerCore extends TrainWorker {

        public HadoopTrainWorkerCore(String modelName,
                                     String configPath,
                                     String configFile,
                                     String pyTransformScript,
                                     boolean needPyTransform,
                                     String loginName,
                                     String hostName,
                                     int hostPort,
                                     int threadNum) throws Exception {
            super(modelName, configPath, configFile, pyTransformScript, needPyTransform, loginName, hostName, hostPort, threadNum);
        }
    }

    public static class TrainMapper extends Mapper<Object, Text, Text, Text> {
        Random rand = new Random();
        @Override
        protected void map(Object key, Text value, Context context) throws IOException, InterruptedException {
            context.write(new Text(rand.nextInt(100000) + ""), value);
        }
    }

    public static class TrainReducer extends Reducer<Text, Text, Text, Text> {
        public static final Logger LOG = LoggerFactory.getLogger(TrainReducer.class);
        public final static LinkedBlockingQueue<String> trainDataQueue = new LinkedBlockingQueue<>(1000000);
        public Thread workThread;
        public static volatile boolean finished = false;
        public int putcnt = 0;

        public static class ReducerIterator implements Iterator<String> {

            @Override
            public boolean hasNext() {
                while(!finished) {
                    if (trainDataQueue.size() > 0) {
                        return true;
                    }
                }

                return trainDataQueue.size() > 0;
            }

            @Override
            public String next() {
                try {
                    String val = trainDataQueue.take();
                    return val;
                } catch (InterruptedException e) {
                    e.printStackTrace();
                    System.exit(1);
                }
                return null;
            }
        }

        @Override
        protected void setup(Context context) {
            Configuration conf = context.getConfiguration();
            workThread = new Thread() {
                @Override
                public void run() {
                    HadoopTrainWorkerCore trainWorkerCore = null;
                    try {
                        trainWorkerCore = new HadoopTrainWorkerCore(
                                conf.get("modelName"),
                                conf.get("configPath"),
                                conf.get("configFile"),
                                conf.get("pyTransformScript"),
                                conf.getBoolean("needPyTransform", false),
                                conf.get("loginName"),
                                conf.get("hostName"),
                                conf.getInt("hostPort", -1),
                                conf.getInt("threadNum", -1)
                                );

                        Map<String, Object> customParamsMap = decodeMap(conf.get("customParamsMap"));

                        for (Map.Entry<String, Object> entry : customParamsMap.entrySet()) {
                            trainWorkerCore.setCustomParam(entry.getKey(), entry.getValue());
                            LOG.info("hadoop custom params:" + entry.getKey() + "=" + entry.getValue());
                        }

                        trainWorkerCore.train(Arrays.asList(new ReducerIterator()), null);
                    } catch (Exception e) {
                        e.printStackTrace();
                        System.exit(1);
                    }
                }
            };
            workThread.start();
        }

        @Override
        protected void cleanup(Context context) throws IOException, InterruptedException {
            finished = true;
            LOG.info("put train data cnt:" + putcnt + ", queue size:" + trainDataQueue.size());
            workThread.join();

        }

        @Override
        public void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
            Iterator<Text> it = values.iterator();
            while (it.hasNext()) {
                String val = it.next().toString();
                trainDataQueue.put(val);
                putcnt ++;
            }
        }
    }

    public static void main(String[] args) throws Exception {

        Configuration conf = new Configuration();
        String[] otherArgs = new GenericOptionsParser(conf, args).getRemainingArgs();
        LOG.info("args:" + Arrays.toString(args));
        LOG.info("other args:" + Arrays.toString(otherArgs));

        String modelName = otherArgs[0];
        String configPath = otherArgs[1];
        String configFile = otherArgs[2];
        String pyTransformScript = otherArgs[3];
        String pyTransformScriptPath = otherArgs[4];
        boolean needPyTransform = Boolean.parseBoolean(otherArgs[5]);
        String loginName = otherArgs[6];
        String hostName = otherArgs[7];
        int hostPort = Integer.parseInt(otherArgs[8]);
        int slaveNum = Integer.parseInt(otherArgs[9]);
        int threadNum = Integer.parseInt(otherArgs[10]);
        String reduceMemM = otherArgs[11];
        String queueName = otherArgs[12];
        String user = otherArgs[13];

        LOG.info("configFile:" + configFile);
        LOG.info("configPath:" + configPath);
        LOG.info("pyTransformScript:" + pyTransformScript);
        LOG.info("pyTransformScriptPath:" + pyTransformScriptPath);
        LOG.info("loginName:" + loginName);
        LOG.info("hostName:" + hostName + ", hostPort:" + hostPort);
        LOG.info("slaveNum:" + slaveNum + ", threadNum:" + threadNum);
        LOG.info("modelName:" + modelName);
        LOG.info("reduce memory:" + reduceMemM);
        LOG.info("queue:" + queueName);

        HadoopTrainWorker worker = new HadoopTrainWorker(conf,
                modelName,
                configPath,
                configFile,
                pyTransformScript,
                needPyTransform,
                loginName,
                hostName,
                hostPort,
                slaveNum,
                threadNum,
                queueName,
                reduceMemM,
                user);

        if (!worker.hadoopTrain()) {
            throw new Exception("hadoop train exception!");
        }

        System.exit(0);
    }
}
