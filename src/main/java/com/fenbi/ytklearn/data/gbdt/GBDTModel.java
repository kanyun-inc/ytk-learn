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

package com.fenbi.ytklearn.data.gbdt;

import com.fenbi.ytklearn.data.Constants;
import com.fenbi.ytklearn.data.Tuple;
import com.fenbi.ytklearn.utils.CheckUtils;

import java.io.BufferedReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.regex.Pattern;

/**
 * @author wufan
 */

public class GBDTModel {

    public float basePrediction;
    public int numTreeInGroup;
    public String objName;
    public List<Tree> trees;

    public GBDTModel() {
        basePrediction = 0.f;
        numTreeInGroup = 1;
        objName = "";
        trees = new ArrayList<>(Constants.RESERVE_NUM);
    }

    public GBDTModel(float basePrediction, int numTreeInGroup, String objName) {
        this.basePrediction = basePrediction;
        this.numTreeInGroup = numTreeInGroup;
        this.objName = objName;
        this.trees = new ArrayList<>(Constants.RESERVE_NUM);
    }

    public List<String> dumpModel(boolean withStats) {
        CheckUtils.check(trees.size() != 0, "GBDT: save model error, tree number is 0!");
        List<String> dump = new ArrayList<>(trees.size() + 1);
        StringBuffer sb = new StringBuffer("");
        sb.append("uniform_base_prediction=" + basePrediction + "\n");
        sb.append("class_num=" + numTreeInGroup + "\n");
        sb.append("loss_function=" + objName + "\n");
        sb.append("tree_num=" + trees.size() + "\n");
        dump.add(sb.toString());

        for (int i = 0; i < trees.size(); i++) {
            dump.add(trees.get(i).dumpModel(i, withStats));
        }
        return dump;
    }

    public void loadModel(BufferedReader reader) throws IOException, ClassNotFoundException {
        CheckUtils.check(trees == null || trees.size() == 0, "GBDT: model has already been initialized!");
        basePrediction = Float.parseFloat(reader.readLine().split("=")[1]);
        numTreeInGroup =  Integer.parseInt(reader.readLine().split("=")[1]);
        objName = reader.readLine().split("=")[1];

        int treeNum = Integer.parseInt(reader.readLine().split("=")[1]);
        CheckUtils.check(treeNum != 0, "GBDT: load model error, tree number is 0!");
        trees = new ArrayList<>(treeNum);

        Pattern innerNodePattern = Pattern.compile(Tree.innerNodePatternStr);
        Pattern leafPattern = Pattern.compile(Tree.leafPatternStr);
        Tree tree;
        for (int i = 0; i < treeNum; i++) {
            tree = new Tree();
            tree.loadModel(reader, innerNodePattern, leafPattern);
            trees.add(tree);
        }
    }

    public Map<String, Integer> genFeatureDict() {
        // generate feature dict from model
        Map<String, Integer> feaDict = new HashMap<>(256);
        for (Tree tree: trees) {
            tree.genFeatureDict(feaDict);
        }
        return feaDict;
    }

    public Map<String, Tuple<Integer, Double>> getFeatureImportance() {
        Map<String, Tuple<Integer, Double>> feaImpMap = new HashMap<>(256);
        for (Tree tree : trees) {
            tree.featureImportance(feaImpMap);
        }
        return feaImpMap;
    }


    @Override
    public String toString() {
        return "global_uniform_base_prediction=" + basePrediction
                + ", obj_name=" + objName
                + ", class_num=" + numTreeInGroup
                + ", tree_num=" + trees.size();

    }
}
