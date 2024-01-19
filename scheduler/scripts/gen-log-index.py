#!/usr/bin/env python3
# import glob
import sys


def main(args):
    for f in args:
        a = "<li><a href=\"./%s\">%s</a></li>" % (f, f)
        print(a)


main(sys.argv[1:])
