# pip3 install --no-index --user -U .

import os
import time


def auto_version():
    major = 0
    minor = 0
    t = os.getenv('GIT_COMMIT_TIMESTAMP')
    patch = str(t) if t else int(time.time())
    return '.'.join([str(x) for x in [major, minor, patch]])


from setuptools import find_packages, setup

setup(
    name='training_state',
    version=auto_version(),
    packages=find_packages(),
    description='',
    url='',
    ext_modules=[],
    setup_requires=[],
    install_requires=[],
)
