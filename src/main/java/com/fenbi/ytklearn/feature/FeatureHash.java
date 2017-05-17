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

package com.fenbi.ytklearn.feature;

import com.fenbi.ytklearn.data.Tuple;
import com.fenbi.ytklearn.utils.NumConvertUtils;
import com.google.common.base.Joiner;
import com.google.common.collect.Lists;
import com.google.common.hash.HashFunction;
import com.google.common.hash.Hashing;
import org.apache.commons.lang.StringUtils;

import java.nio.charset.Charset;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * @author xialong
 */

public class FeatureHash {
    private final int hashBucketSize;
    private final String hashFeaturePrefix;
    private final HashFunction murmurHash;

    private String featuresDelim;
    private String featureNameValueDelim;

    private Charset charset = Charset.defaultCharset();


    Joiner.MapJoiner joiner;

    private FeatureHash(int hashBucketSize,
                        int hashSeed,
                        String hashFeaturePrefix) {
        this.hashBucketSize = hashBucketSize;
        this.hashFeaturePrefix = hashFeaturePrefix;
        this.murmurHash =  Hashing.murmur3_128(hashSeed);
    }

    public FeatureHash withDelim(String featuresDelim, String featureNameValueDelim) {
        this.featuresDelim = featuresDelim;
        this.featureNameValueDelim = featureNameValueDelim;
        joiner = Joiner.on(featuresDelim).withKeyValueSeparator(featureNameValueDelim);
        return this;
    }

//    public final int hash(String val) {
//        return (int)(murmurHash.hashString(val,
//                charset).asLong() & 0x7fffffffL);
//    }
//
//
//    public int signhash(String val) {
//        return (int)((murmurHash.hashString(val,
//                charset).asLong() & 0x10000000000L) >> 40) - 1;
//    }

    public Map<String, Float> line2FeatureMap(String line) {
        Map<String, Float> X = new HashMap<>();
        String [] finfo = line.trim().split(featuresDelim);
        for (String f : finfo) {
            String []fvinfo = f.split(featureNameValueDelim);
            X.put(fvinfo[0].trim(), NumConvertUtils.parseFloat(fvinfo[1]));
        }

        return X;
    }

    public Map<String, Float> hashMap2Map(Map<String, Float> XMap) {
        Map<String, Float> fhashMap = new HashMap<>(XMap.size());

        for (Map.Entry<String, Float> x : XMap.entrySet()) {

            long longhash = murmurHash.hashString(x.getKey(),
                    charset).asLong();
            int fhash = (int)((longhash & 0x7fffffffL) % hashBucketSize);
            float signhash = 2.0f * ((longhash & 0x10000000000L) >> 40) - 1.0f;

            //System.out.println("fhash:" + fhash + ", sign:" + signhash);

            String fname = hashFeaturePrefix + fhash;
            Float oldval = fhashMap.get(fname);
            if (oldval == null) {
                fhashMap.put(fname, signhash * x.getValue());
            } else {
                fhashMap.put(fname, oldval.floatValue() + signhash * x.getValue());
            }
        }

        return fhashMap;
    }

    public Map<String, Float> hashPairList2Map(List<Tuple<String, Float>> XList) {
        Map<String, Float> fhashMap = new HashMap<>(XList.size());

        for (Tuple<String, Float> x : XList) {

            long longhash = murmurHash.hashString(x.v1,
                    charset).asLong();
            int fhash = (int)((longhash & 0x7fffffffL) % hashBucketSize);
            float signhash = 2.0f * ((longhash & 0x10000000000L) >> 40) - 1.0f;

            //System.out.println("fhash:" + fhash + ", sign:" + signhash);

            String fname = hashFeaturePrefix + fhash;
            Float oldval = fhashMap.get(fname);
            if (oldval == null) {
                fhashMap.put(fname, signhash * x.v2);
            } else {
                fhashMap.put(fname, oldval.floatValue() + signhash * x.v2);
            }
        }

        return fhashMap;
    }

    public Map<String, Float> hashList2Map(List<String> XList) {
        Map<String, Float> fhashMap = new HashMap<>(XList.size());
        for (String x : XList) {

            long longhash = murmurHash.hashString(x,
                    charset).asLong();
            int fhash = (int)((longhash & 0x7fffffffL) % hashBucketSize);
            float signhash = 2.0f * ((longhash & 0x10000000000L) >> 40) - 1.0f;

            //System.out.println("fhash:" + fhash + ", sign:" + signhash);

            String fname = hashFeaturePrefix + fhash;
            Float oldval = fhashMap.get(fname);
            if (oldval == null) {
                fhashMap.put(fname, signhash);
            } else {
                fhashMap.put(fname, oldval.floatValue() + signhash);
            }
        }

        return fhashMap;
    }


    public Map<String, Float> hashLine2Map(String line) {
        Map<String, Float> X = new HashMap<>();
        String [] finfo = line.trim().split(featuresDelim);
        for (String f : finfo) {
            String []fvinfo = f.split(featureNameValueDelim);
            X.put(fvinfo[0].trim(), NumConvertUtils.parseFloat(fvinfo[1]));
        }

        return hashMap2Map(X);
    }

    public Map<String, Float> line2Map(String line, boolean needFeatureHash) {
        if (needFeatureHash) {
            return hashLine2Map(line);
        } else {
            return line2FeatureMap(line);
        }
    }



    public String hashMap2Line(Map<String, Float> XMap) {
        return joiner.join(hashMap2Map(XMap));
    }

    public String hashList2Line(List<String> XList) {
        return joiner.join(hashList2Map(XList));
    }

    public String hashLine2Line(String line) {
        return joiner.join(hashLine2Map(line));
    }


    public static FeatureHash build(int hashBucketSize,
                                    int hashSeed,
                                    String hashFeaturePrefix) {
        return new FeatureHash(hashBucketSize, hashSeed, hashFeaturePrefix);
    }


    public static void main(String []args) {
        FeatureHash featureHash = FeatureHash.build(10, 111311, "hash_").withDelim(",", ":");
        String featureline = "1:1,2:1,3:1,4:1,5:1,6:1,7:1,8:1,9:1";
        List<String> flist = Lists.newArrayList(StringUtils.split("1,2,3,4,5,6,7,8,9", ","));
        List<Tuple<String, Float>> ftlist = new ArrayList<>();
        for (int i = 1; i <= 9; i++) {
            ftlist.add(new Tuple<>(i + "", 1.0f));
        }
        System.out.println("feature line:" + featureline);
        System.out.println("hash map 2 map:" + featureHash.hashMap2Map(parse(featureline, ",", ":")));
        System.out.println("hash map 2 map:" + featureHash.hashMap2Map(parse(featureline, ",", ":")));

        System.out.println("hash list 2 map:" + featureHash.hashList2Map(flist));
        System.out.println("hash list 2 map:" + featureHash.hashList2Map(flist));

        System.out.println("hash tuple list 2 map:" + featureHash.hashPairList2Map(ftlist));
        System.out.println("hash tuple list 2 map:" + featureHash.hashPairList2Map(ftlist));

        System.out.println("hash line 2 map:" + featureHash.hashLine2Map(featureline));
        System.out.println("hash line 2 map:" + featureHash.hashLine2Map(featureline));

        System.out.println("hash map 2 line:" + featureHash.hashMap2Line(parse(featureline, ",", ":")));
        System.out.println("hash map 2 line:" + featureHash.hashMap2Line(parse(featureline, ",", ":")));

        System.out.println("hash list 2 line:" + featureHash.hashList2Line(flist));
        System.out.println("hash list 2 line:" + featureHash.hashList2Line(flist));

        System.out.println("hash line 2 line:" + featureHash.hashLine2Line(featureline));
        System.out.println("hash line 2 line:" + featureHash.hashLine2Line(featureline));



        int time = 10000000;
        int size = 10;
        long start = System.currentTimeMillis();
        int cnt[] = new int[size];
        int signcnt[] = new int[2];

        HashFunction hashFunction = Hashing.murmur3_128(111);
        for (int i = 0; i < time; i++) {
//            cnt[featureHash.hash("fsfdsfdsf" + i) % size] ++;
//            signcnt[featureHash.signhash("fsfdsfdsf" + i) + 1] ++;

            long longhash = hashFunction.hashString("fsfdsfdsf" + i,
                    Charset.defaultCharset()).asLong();
            int fhash = (int)((longhash & 0x7fffffffL) % size);
            float signhash = 2.0f * ((longhash & 0x10000000000L) >> 40) - 1.0f;
            cnt[fhash] ++;
            int sign = 0;
            if (signhash < 0) {
                sign = 0;
            } else {
                sign = 1;
            }
            signcnt[sign] ++;
        }
        long end = System.currentTimeMillis();
        System.out.println("takes:" + time * 1000. / (end - start));

        for (int i = 0; i < size; i++) {
            System.out.println(i + ":" + cnt[i] * 1.0 / time);
        }

        System.out.println(signcnt[0] * 1.0 / time);
        System.out.println(signcnt[1] * 1.0 / time);

//        for (int i = 0; i < 20; i++) {
//            System.out.println(featureHash.hash("xx"));
//            System.out.println(featureHash.hash("yy"));
//        }
    }

    public static Map<String, Float> parse(String line, String featuresDelim, String featureNameValueDelim) {
        Map<String, Float> fmap = new HashMap<>();
        String []info = line.trim().split(",");
        for (String f : info) {
            String []fv = f.split(":");
            fmap.put(fv[0].trim(), Float.parseFloat(fv[1]));
        }

        return fmap;
    }

}
