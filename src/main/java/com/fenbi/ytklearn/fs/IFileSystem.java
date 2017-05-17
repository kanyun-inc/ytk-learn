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

package com.fenbi.ytklearn.fs;

import java.io.*;
import java.util.Iterator;
import java.util.List;

/**
 * @author xialong
 */

public interface IFileSystem {
    public boolean exists(String path) throws IOException;
    public Reader getReader(String path) throws IOException;
    public Writer getWriter(String path) throws IOException;
    public InputStream getInputStream(String path) throws IOException;
    public OutputStream getOutputStream(String path) throws IOException;
    public List<String> recurGetPaths(List<String> paths) throws IOException;
    public List<Iterator<String>> read(List<String> paths) throws IOException;
    public List<Iterator<String>> selectRead(List<String> paths, int divisor, int remainer) throws IOException;
    public void delete(String path) throws IOException;
    public void mkdirs(String path) throws IOException;
}
