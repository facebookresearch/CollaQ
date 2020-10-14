import os
import sys
header = """Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
def prefixed_header(prefix):
    return [ prefix + l + "\n" for l in header.split("\n")]
filename = sys.argv[1]
with open(filename, "r") as f:
    lines = f.readlines()
ext = os.path.splitext(filename)[1]
if ext in (".c", ".cc", ".h", ".cpp"):
    lines = prefixed_header("// ") + lines
elif ext in (".py"):
    lines = prefixed_header("# ") + lines
else:
    print(f"File {filename} is not python or C/C++. Skip..")
    sys.exit(0)
with open(filename, "w") as f:
    f.writelines(lines)