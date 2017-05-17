#!/user/bin/env python
# -*- coding: UTF-8 -*-

# this script used for kill yarn jobs, used by spark/hadoop_optimizer.sh,
# usage: python yarn_job_kill.py yarn_log_file

import re
import sys
import os

if __name__ == "__main__":
    yarn_log_path = sys.argv[1]
    reader = open(yarn_log_path, "r")
    p = re.compile(r"(application_\d+_\d+)")
    id_set = set()
    id_list = []
    for line in reader:
        s = p.search(line)
        if (s == None):
            continue
        sid = s.group(1)
        # print sid
        if sid not in id_set:
            id_set.add(sid)
            id_list.append(sid)

    # print id_list
    for i in range(max(-8, -len(id_list)), 0):
        sid = id_list[i]
        print "last application id:" + sid
        kill = "yarn application -kill " + sid
        print kill
        os.system(kill)
