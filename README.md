<div align="center">
  <img src="https://www.tensorflow.org/images/tf_logo_transp.png"><br><br>
</div>
-----------------

|  **`Linux CPU`**   |  **`Linux GPU PIP`** | **`Mac OS CPU`** |  **`Android`** |
|-------------------|----------------------|------------------|----------------|
| [![Build Status](http://ci.tensorflow.org/buildStatus/icon?job=tensorflow-master-cpu)](http://ci.tensorflow.org/job/tensorflow-master-cpu) | [![Build Status](http://ci.tensorflow.org/buildStatus/icon?job=tensorflow-master-gpu_pip)](http://ci.tensorflow.org/job/tensorflow-master-gpu_pip) | [![Build Status](http://ci.tensorflow.org/buildStatus/icon?job=tensorflow-master-mac)](http://ci.tensorflow.org/job/tensorflow-master-mac) | [![Build Status](http://ci.tensorflow.org/buildStatus/icon?job=tensorflow-master-android)](http://ci.tensorflow.org/job/tensorflow-master-android) |

**TensorFlow** is an open source software library for numerical computation using
data flow graphs.  Nodes in the graph represent mathematical operations, while
the graph edges represent the multidimensional data arrays (tensors) that flow
between them.  This flexible architecture lets you deploy computation to one
or more CPUs or GPUs in a desktop, server, or mobile device without rewriting
code.  TensorFlow was originally developed by researchers and engineers
working on the Google Brain team within Google's Machine Intelligence research
organization for the purposes of conducting machine learning and deep neural
networks research.  The system is general enough to be applicable in a wide
variety of other domains, as well.

**If you'd like to contribute to tensorflow, be sure to review the [contribution
guidelines](CONTRIBUTING.md).**

**We use [github issues](https://github.com/tensorflow/tensorflow/issues) for
tracking requests and bugs, but please see
[Community](tensorflow/g3doc/resources/index.md#community) for general questions
and discussion.**

## Installation
*See [Download and Setup](tensorflow/g3doc/get_started/os_setup.md) for instructions on how to install our release binaries or how to build from source.*

People who are a little bit adventurous can also try our nightly binaries:

* Linux CPU only: [Python 2](http://ci.tensorflow.org/view/Nightly/job/nightly-matrix-cpu/TF_BUILD_CONTAINER_TYPE=CPU,TF_BUILD_IS_OPT=OPT,TF_BUILD_IS_PIP=PIP,TF_BUILD_PYTHON_VERSION=PYTHON2,label=cpu-slave/lastSuccessfulBuild/artifact/pip_test/whl/tensorflow-0.7.1-cp27-none-linux_x86_64.whl) ([build history](http://ci.tensorflow.org/view/Nightly/job/nightly-matrix-cpu/TF_BUILD_CONTAINER_TYPE=CPU,TF_BUILD_IS_OPT=OPT,TF_BUILD_IS_PIP=PIP,TF_BUILD_PYTHON_VERSION=PYTHON2,label=cpu-slave/)) / [Python 3](http://ci.tensorflow.org/view/Nightly/job/nightly-matrix-cpu/TF_BUILD_CONTAINER_TYPE=CPU,TF_BUILD_IS_OPT=OPT,TF_BUILD_IS_PIP=PIP,TF_BUILD_PYTHON_VERSION=PYTHON3,label=cpu-slave/lastSuccessfulBuild/artifact/pip_test/whl/tensorflow-0.7.1-cp34-cp34m-linux_x86_64.whl) ([build history](http://ci.tensorflow.org/view/Nightly/job/nightly-matrix-cpu/TF_BUILD_CONTAINER_TYPE=CPU,TF_BUILD_IS_OPT=OPT,TF_BUILD_IS_PIP=PIP,TF_BUILD_PYTHON_VERSION=PYTHON3,label=cpu-slave/))
* Linux GPU: [Python 2](http://ci.tensorflow.org/view/Nightly/job/nigntly-matrix-linux-gpu/TF_BUILD_CONTAINER_TYPE=GPU,TF_BUILD_IS_OPT=OPT,TF_BUILD_IS_PIP=PIP,TF_BUILD_PYTHON_VERSION=PYTHON2,label=gpu-slave/lastSuccessfulBuild/artifact/pip_test/whl/tensorflow-0.7.1-cp27-none-linux_x86_64.whl) ([build history](http://ci.tensorflow.org/view/Nightly/job/nigntly-matrix-linux-gpu/TF_BUILD_CONTAINER_TYPE=GPU,TF_BUILD_IS_OPT=OPT,TF_BUILD_IS_PIP=PIP,TF_BUILD_PYTHON_VERSION=PYTHON2,label=gpu-slave/)) / [Python 3](http://ci.tensorflow.org/view/Nightly/job/nigntly-matrix-linux-gpu/TF_BUILD_CONTAINER_TYPE=GPU,TF_BUILD_IS_OPT=OPT,TF_BUILD_IS_PIP=PIP,TF_BUILD_PYTHON_VERSION=PYTHON3,label=gpu-slave/lastSuccessfulBuild/artifact/pip_test/whl/tensorflow-0.7.1-cp34-cp34m-linux_x86_64.whl) ([build history](http://ci.tensorflow.org/view/Nightly/job/nigntly-matrix-linux-gpu/TF_BUILD_CONTAINER_TYPE=GPU,TF_BUILD_IS_OPT=OPT,TF_BUILD_IS_PIP=PIP,TF_BUILD_PYTHON_VERSION=PYTHON3,label=gpu-slave/))
* Mac CPU only: [Python 2](http://ci.tensorflow.org/view/Nightly/job/nightly-matrix-cpu/TF_BUILD_CONTAINER_TYPE=CPU,TF_BUILD_IS_OPT=OPT,TF_BUILD_IS_PIP=PIP,TF_BUILD_PYTHON_VERSION=PYTHON2,label=mac-slave/lastSuccessfulBuild/artifact/pip_test/whl/tensorflow-0.7.1-py2-none-any.whl) ([build history](http://ci.tensorflow.org/view/Nightly/job/nightly-matrix-cpu/TF_BUILD_CONTAINER_TYPE=CPU,TF_BUILD_IS_OPT=OPT,TF_BUILD_IS_PIP=PIP,TF_BUILD_PYTHON_VERSION=PYTHON2,label=mac-slave/)) / [Python 3](http://ci.tensorflow.org/view/Nightly/job/nightly-matrix-cpu/TF_BUILD_CONTAINER_TYPE=CPU,TF_BUILD_IS_OPT=OPT,TF_BUILD_IS_PIP=PIP,TF_BUILD_PYTHON_VERSION=PYTHON3,label=mac-slave/lastSuccessfulBuild/artifact/pip_test/whl/tensorflow-0.7.1-py3-none-any.whl) ([build history](http://ci.tensorflow.org/view/Nightly/job/nightly-matrix-cpu/TF_BUILD_CONTAINER_TYPE=CPU,TF_BUILD_IS_OPT=OPT,TF_BUILD_IS_PIP=PIP,TF_BUILD_PYTHON_VERSION=PYTHON3,label=mac-slave/))
* [Android](http://ci.tensorflow.org/view/Nightly/job/nightly-matrix-android/TF_BUILD_CONTAINER_TYPE=ANDROID,TF_BUILD_IS_OPT=OPT,TF_BUILD_IS_PIP=NO_PIP,TF_BUILD_PYTHON_VERSION=PYTHON2,label=android-slave/lastSuccessfulBuild/artifact/bazel-out/local_linux/bin/tensorflow/examples/android/tensorflow_demo.apk) ([build history](http://ci.tensorflow.org/view/Nightly/job/nightly-matrix-android/TF_BUILD_CONTAINER_TYPE=ANDROID,TF_BUILD_IS_OPT=OPT,TF_BUILD_IS_PIP=NO_PIP,TF_BUILD_PYTHON_VERSION=PYTHON2,label=android-slave/))

#### *Try your first TensorFlow program*
```python
$ python

>>> import tensorflow as tf
>>> hello = tf.constant('Hello, TensorFlow!')
>>> sess = tf.Session()
>>> sess.run(hello)
Hello, TensorFlow!
>>> a = tf.constant(10)
>>> b = tf.constant(32)
>>> sess.run(a+b)
42
>>>
```

##For more information

* [TensorFlow website](http://tensorflow.org)
* [TensorFlow whitepaper](http://download.tensorflow.org/paper/whitepaper2015.pdf)
* [TensorFlow MOOC on Udacity] (https://www.udacity.com/course/deep-learning--ud730)

The TensorFlow community has created amazing things with TensorFlow, please see the [resources section of tensorflow.org](https://www.tensorflow.org/versions/master/resources#community) for an incomplete list.
