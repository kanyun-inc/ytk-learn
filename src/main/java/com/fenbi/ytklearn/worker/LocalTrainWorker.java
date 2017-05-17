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

/**
 * @author xialong
 */

public class LocalTrainWorker extends TrainWorker {
    public LocalTrainWorker(String modelName,
                            String configPath,
                            String configFile,
                            String pyTransformScript,
                            boolean needPyTransform,
                            String loginName,
                            String hostName,
                            int hostPort,
                            int threadNum) throws Exception {
        super(modelName, configPath, configFile, pyTransformScript, needPyTransform,
                loginName, hostName, hostPort, threadNum);
    }

    public boolean localTrain() throws Exception {
        return train(null, null);
    }


    public static void main(String []args) {

        String modelName = args[0];
        String configPath = args[1];
        String configFile = configPath;
        String pyTransformScript = args[2];
        boolean needPyTransform = Boolean.parseBoolean(args[3]);
        String loginName = args[4];
        String hostName = args[5];
        int hostPort = Integer.parseInt(args[6]);
        int threadNum = Integer.parseInt(args[7]);

        try {
            LocalTrainWorker worker = new LocalTrainWorker(
                    modelName,
                    configPath,
                    configFile,
                    pyTransformScript,
                    needPyTransform,
                    loginName,
                    hostName,
                    hostPort,
                    threadNum);
            if (!worker.localTrain()) {
                throw new Exception("local train failed!");
            }
        } catch (Exception e) {
            LOG.error("local train exception!", e);
            System.exit(1);
        }

        System.exit(0);
    }
}
