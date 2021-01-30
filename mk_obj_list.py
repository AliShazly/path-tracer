#!/usr/bin/env python3

import os
import sys

if len(sys.argv) < 2:
    sys.stderr.write("Err: specify a path\n")
    exit(1)

obj_dir = os.path.abspath(sys.argv[1])

with open("objs.txt", "w") as f:
    for obj_path in os.listdir(obj_dir):
        write_path = os.path.join(obj_dir, obj_path);
        f.write(f"{write_path}\n")
