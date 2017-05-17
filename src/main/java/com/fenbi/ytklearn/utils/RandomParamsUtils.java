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

package com.fenbi.ytklearn.utils;

import com.fenbi.ytklearn.param.RandomParams;
import com.typesafe.config.Config;
import com.typesafe.config.ConfigFactory;

import java.io.File;
import java.util.Random;

/**
 * @author xialong
 */

public final class RandomParamsUtils {
    private final Random rand;
    private final RandomParams randomParams;

    public RandomParamsUtils(RandomParams randomParams) {
        rand = new Random(randomParams.seed);
        this.randomParams = randomParams;
    }

    public RandomParamsUtils(RandomParams randomParams, int seed) {
        rand = new Random(seed);
        this.randomParams = randomParams;
    }

    public double nextNormal() {
        return rand.nextGaussian() * randomParams.normal.std + randomParams.normal.mean;
    }

    public double nextUniform() {
        return randomParams.uniform.range_start +
                (randomParams.uniform.range_end - randomParams.uniform.range_start) * rand.nextDouble();
    }

    public double nextNormal(double mean, double std) {
        return rand.nextGaussian() * std + mean;
    }

    public double nextUniform(double rangeStart, double rangeEnd) {
        return rangeStart + (rangeEnd - rangeStart) * rand.nextDouble();
    }

    public double next() {
        if (randomParams.mode == RandomParams.Mode.NORMAL) {
            return nextNormal();
        } else {
            return nextUniform();
        }
    }

    public double next01() {
        return rand.nextDouble();
    }

    public static void main(String []args) {
        Config config = ConfigFactory.parseFile(new File("config/model/fm.conf"));
        RandomParams randomParams = new RandomParams(config, "");
        RandomParamsUtils randomParamsUtils = new RandomParamsUtils(randomParams);

        System.out.println("normal sample:");
        for (int i = 0; i < 50; i++) {
            System.out.println(randomParamsUtils.next());
        }

        System.out.println("uniform sample:");
        for (int i = 0; i < 50000; i++) {
            double r = randomParamsUtils.next();
            if (r < -0.01 || r > 0.01) {
                System.out.println("error");
                break;
            }
        }
    }

}
