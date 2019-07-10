# TensorFlow Lite Micro Mbed Project

This folder has been autogenerated by TensorFlow, and contains source, header,
and project files needed to build a single TensorFlow Lite Micro target using
the Mbed command line interface.

## Usage

To load the dependencies this code requires, run:

```
mbed config root .
mbed deploy
```

TensorFlow requires C++ 11, so you'll need to update your profiles to reflect
this. Here's a short Python command that does that:

```
python -c 'import fileinput, glob;
for filename in glob.glob("mbed-os/tools/profiles/*.json"):
  for line in fileinput.input(filename, inplace=True):
    print line.replace("\"-std=gnu++98\"","\"-std=c++11\", \"-fpermissive\"")'
```

With that setting updated, you should now be able to compile:

```
mbed compile -m auto -t GCC_ARM
```

If this works, it will give you a .bin file that you can flash onto the device
you're targeting. For example, using a Discovery STM3246G board, you can deploy
it by copying the bin to the volume mounted as a USB drive, just by dragging
over the file.

## Project Generation

See
[tensorflow/lite/experimental/micro](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/experimental/micro)
for details on how projects like this can be generated from the main source
tree.

## License

TensorFlow's code is covered by the Apache2 License included in the repository,
and third party dependencies are covered by their respective licenses, in the
third_party folder of this package.
