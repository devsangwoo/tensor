# TensorFlow external dependencies that can be loaded in WORKSPACE files.

load("//third_party/gpus:cuda_configure.bzl", "cuda_configure")
load("//third_party/mkl:build_defs.bzl", "mkl_repository")
load("//third_party/py:python_configure.bzl", "python_configure")
load("//third_party/sycl:sycl_configure.bzl", "sycl_configure")
load("//third_party/toolchains/cpus/arm:arm_compiler_configure.bzl", "arm_compiler_configure")
load("//third_party:repo.bzl", "tf_http_archive")
load("@io_bazel_rules_closure//closure/private:java_import_external.bzl", "java_import_external")
load("@io_bazel_rules_closure//closure:defs.bzl", "filegroup_external")

# Parse the bazel version string from `native.bazel_version`.
def _parse_bazel_version(bazel_version):
  # Remove commit from version.
  version = bazel_version.split(" ", 1)[0]
  # Split into (release, date) parts and only return the release
  # as a tuple of integers.
  parts = version.split("-", 1)
  # Turn "release" into a tuple of strings
  version_tuple = ()
  for number in parts[0].split("."):
    version_tuple += (str(number),)
  return version_tuple

# Check that a specific bazel version is being used.
def check_version(bazel_version):
  if "bazel_version" not in dir(native):
    fail("\nCurrent Bazel version is lower than 0.2.1, expected at least %s\n" %
         bazel_version)
  elif not native.bazel_version:
    print("\nCurrent Bazel is not a release version, cannot check for " +
          "compatibility.")
    print("Make sure that you are running at least Bazel %s.\n" % bazel_version)
  else:
    current_bazel_version = _parse_bazel_version(native.bazel_version)
    minimum_bazel_version = _parse_bazel_version(bazel_version)
    if minimum_bazel_version > current_bazel_version:
      fail("\nCurrent Bazel version is {}, expected at least {}\n".format(
          native.bazel_version, bazel_version))

# If TensorFlow is linked as a submodule.
# path_prefix is no longer used.
# tf_repo_name is thought to be under consideration.
def tf_workspace(path_prefix="", tf_repo_name=""):
  # We must check the bazel version before trying to parse any other BUILD
  # files, in case the parsing of those build files depends on the bazel
  # version we require here.
  check_version("0.5.4")
  cuda_configure(name="local_config_cuda")
  sycl_configure(name="local_config_sycl")
  python_configure(name="local_config_python")

  # Point //external/local_config_arm_compiler to //external/arm_compiler
  arm_compiler_configure(
      name="local_config_arm_compiler",
      remote_config_repo="../arm_compiler",
      build_file = str(Label("//third_party/toolchains/cpus/arm:BUILD")))

  mkl_repository(
      name = "mkl",
      urls = [
          "https://mirror.bazel.build/github.com/01org/mkl-dnn/releases/download/v0.9/mklml_lnx_2018.0.20170720.tgz",
          "https://github.com/01org/mkl-dnn/releases/download/v0.9/mklml_lnx_2018.0.20170720.tgz",
      ],
      sha256 = "57ba56c4c243f403ff78f417ff854ef50b9eddf4a610a917b7c95e7fa8553a4b",
      strip_prefix = "mklml_lnx_2018.0.20170720",
      build_file = str(Label("//third_party/mkl:mkl.BUILD")),
  )

  if path_prefix:
    print("path_prefix was specified to tf_workspace but is no longer used " +
          "and will be removed in the future.")

  tf_http_archive(
      name = "mkl_dnn",
      urls = [
          "https://mirror.bazel.build/github.com/01org/mkl-dnn/archive/aab753280e83137ba955f8f19d72cb6aaba545ef.tar.gz",
          "https://github.com/01org/mkl-dnn/archive/aab753280e83137ba955f8f19d72cb6aaba545ef.tar.gz",
      ],
      sha256 = "fb67f255a96bd4ad39b8dd104eca5aa92200c95c1ed36e59641e6c0478eefd11",
      strip_prefix = "mkl-dnn-aab753280e83137ba955f8f19d72cb6aaba545ef",
      build_file = str(Label("//third_party/mkl_dnn:mkldnn.BUILD")),
  )

  tf_http_archive(
      name = "com_google_absl",
      urls = [
          "https://mirror.bazel.build/github.com/abseil/abseil-cpp/archive/cc4bed2d74f7c8717e31f9579214ab52a9c9c610.tar.gz",
          "https://github.com/abseil/abseil-cpp/archive/cc4bed2d74f7c8717e31f9579214ab52a9c9c610.tar.gz",
      ],
     sha256 = "f1a7349f88d2846210c42e2f7271dabeee404c2a3b4198e34a797993e3569b03",
     strip_prefix = "abseil-cpp-cc4bed2d74f7c8717e31f9579214ab52a9c9c610",
  )

  tf_http_archive(
      name = "eigen_archive",
      urls = [
          "https://mirror.bazel.build/bitbucket.org/eigen/eigen/get/b6e6d0cf6a77.tar.gz",
          "https://bitbucket.org/eigen/eigen/get/b6e6d0cf6a77.tar.gz",
      ],
      sha256 = "0840c497f2749b5e90bda666aab96be6da90dc75b4e21ca9843cae69b7fed52a",
      strip_prefix = "eigen-eigen-b6e6d0cf6a77",
      build_file = str(Label("//third_party:eigen.BUILD")),
  )

  tf_http_archive(
      name = "arm_compiler",
      sha256 = "970285762565c7890c6c087d262b0a18286e7d0384f13a37786d8521773bc969",
      strip_prefix = "tools-0e906ebc527eab1cdbf7adabff5b474da9562e9f/arm-bcm2708/arm-rpi-4.9.3-linux-gnueabihf",
      urls = [
          "https://mirror.bazel.build/github.com/raspberrypi/tools/archive/0e906ebc527eab1cdbf7adabff5b474da9562e9f.tar.gz",
          # Please uncomment me, when the next upgrade happens. Then
          # remove the whitelist entry in third_party/repo.bzl.
          # "https://github.com/raspberrypi/tools/archive/0e906ebc527eab1cdbf7adabff5b474da9562e9f.tar.gz",
      ],
      build_file = str(Label("//:arm_compiler.BUILD")),
  )

  tf_http_archive(
      name = "libxsmm_archive",
      urls = [
          "https://mirror.bazel.build/github.com/hfp/libxsmm/archive/1.8.1.tar.gz",
          "https://github.com/hfp/libxsmm/archive/1.8.1.tar.gz",
      ],
      sha256 = "2ade869c3f42f23b5263c7d594aa3c7e5e61ac6a3afcaf5d6e42899d2a7986ce",
      strip_prefix = "libxsmm-1.8.1",
      build_file = str(Label("//third_party:libxsmm.BUILD")),
  )

  tf_http_archive(
      name = "ortools_archive",
      urls = [
          "https://mirror.bazel.build/github.com/google/or-tools/archive/253f7955c6a1fd805408fba2e42ac6d45b312d15.tar.gz",
          # Please uncomment me, when the next upgrade happens. Then
          # remove the whitelist entry in third_party/repo.bzl.
          # "https://github.com/google/or-tools/archive/253f7955c6a1fd805408fba2e42ac6d45b312d15.tar.gz",
      ],
      sha256 = "932075525642b04ac6f1b50589f1df5cd72ec2f448b721fd32234cf183f0e755",
      strip_prefix = "or-tools-253f7955c6a1fd805408fba2e42ac6d45b312d15/src",
      build_file = str(Label("//third_party:ortools.BUILD")),
  )

  tf_http_archive(
      name = "com_googlesource_code_re2",
      urls = [
          "https://mirror.bazel.build/github.com/google/re2/archive/26cd968b735e227361c9703683266f01e5df7857.tar.gz",
          "https://github.com/google/re2/archive/26cd968b735e227361c9703683266f01e5df7857.tar.gz",

      ],
      sha256 = "e57eeb837ac40b5be37b2c6197438766e73343ffb32368efea793dfd8b28653b",
      strip_prefix = "re2-26cd968b735e227361c9703683266f01e5df7857",
  )

  tf_http_archive(
      name = "gemmlowp",
      urls = [
          "https://mirror.bazel.build/github.com/google/gemmlowp/archive/010bb3e71a26ca1d0884a167081d092b43563996.zip",
          "https://github.com/google/gemmlowp/archive/010bb3e71a26ca1d0884a167081d092b43563996.zip",
      ],
      sha256 = "dd2557072bde12141419cb8320a9c25e6ec41a8ae53c2ac78c076a347bb46d9d",
      strip_prefix = "gemmlowp-010bb3e71a26ca1d0884a167081d092b43563996",
  )

  tf_http_archive(
      name = "farmhash_archive",
      urls = [
          "https://mirror.bazel.build/github.com/google/farmhash/archive/816a4ae622e964763ca0862d9dbd19324a1eaf45.tar.gz",
          "https://github.com/google/farmhash/archive/816a4ae622e964763ca0862d9dbd19324a1eaf45.tar.gz",
      ],
      sha256 = "6560547c63e4af82b0f202cb710ceabb3f21347a4b996db565a411da5b17aba0",
      strip_prefix = "farmhash-816a4ae622e964763ca0862d9dbd19324a1eaf45",
      build_file = str(Label("//third_party:farmhash.BUILD")),
  )

  tf_http_archive(
      name = "highwayhash",
      urls = [
          "https://mirror.bazel.build/github.com/google/highwayhash/archive/dfcb97ca4fe9277bf9dc1802dd979b071896453b.tar.gz",
          "https://github.com/google/highwayhash/archive/dfcb97ca4fe9277bf9dc1802dd979b071896453b.tar.gz",
      ],
      sha256 = "0f30a15b1566d93f146c8d149878a06e91d9bb7ec2cfd76906df62a82be4aac9",
      strip_prefix = "highwayhash-dfcb97ca4fe9277bf9dc1802dd979b071896453b",
      build_file = str(Label("//third_party:highwayhash.BUILD")),
  )

  tf_http_archive(
      name = "nasm",
      urls = [
          "https://mirror.bazel.build/www.nasm.us/pub/nasm/releasebuilds/2.12.02/nasm-2.12.02.tar.bz2",
          "http://pkgs.fedoraproject.org/repo/pkgs/nasm/nasm-2.12.02.tar.bz2/d15843c3fb7db39af80571ee27ec6fad/nasm-2.12.02.tar.bz2",
      ],
      sha256 = "00b0891c678c065446ca59bcee64719d0096d54d6886e6e472aeee2e170ae324",
      strip_prefix = "nasm-2.12.02",
      build_file = str(Label("//third_party:nasm.BUILD")),
  )

  tf_http_archive(
      name = "jpeg",
      urls = [
          "https://mirror.bazel.build/github.com/libjpeg-turbo/libjpeg-turbo/archive/1.5.1.tar.gz",
          "https://github.com/libjpeg-turbo/libjpeg-turbo/archive/1.5.1.tar.gz",
      ],
      sha256 = "c15a9607892113946379ccea3ca8b85018301b200754f209453ab21674268e77",
      strip_prefix = "libjpeg-turbo-1.5.1",
      build_file = str(Label("//third_party/jpeg:jpeg.BUILD")),
  )

  tf_http_archive(
      name = "png_archive",
      urls = [
          "https://mirror.bazel.build/github.com/glennrp/libpng/archive/v1.2.53.tar.gz",
          "https://github.com/glennrp/libpng/archive/v1.2.53.tar.gz",
      ],
      sha256 = "716c59c7dfc808a4c368f8ada526932be72b2fcea11dd85dc9d88b1df1dfe9c2",
      strip_prefix = "libpng-1.2.53",
      build_file = str(Label("//third_party:png.BUILD")),
  )

  tf_http_archive(
      name = "sqlite_archive",
      urls = [
          "https://mirror.bazel.build/www.sqlite.org/2017/sqlite-amalgamation-3200000.zip",
          "http://www.sqlite.org/2017/sqlite-amalgamation-3200000.zip",
      ],
      sha256 = "208780b3616f9de0aeb50822b7a8f5482f6515193859e91ed61637be6ad74fd4",
      strip_prefix = "sqlite-amalgamation-3200000",
      build_file = str(Label("//third_party:sqlite.BUILD")),
  )

  tf_http_archive(
      name = "gif_archive",
      urls = [
          "https://mirror.bazel.build/ufpr.dl.sourceforge.net/project/giflib/giflib-5.1.4.tar.gz",
          "http://pilotfiber.dl.sourceforge.net/project/giflib/giflib-5.1.4.tar.gz",
      ],
      sha256 = "34a7377ba834397db019e8eb122e551a49c98f49df75ec3fcc92b9a794a4f6d1",
      strip_prefix = "giflib-5.1.4",
      build_file = str(Label("//third_party:gif.BUILD")),
  )

  tf_http_archive(
      name = "six_archive",
      urls = [
          "https://mirror.bazel.build/pypi.python.org/packages/source/s/six/six-1.10.0.tar.gz",
          "https://pypi.python.org/packages/source/s/six/six-1.10.0.tar.gz",
      ],
      sha256 = "105f8d68616f8248e24bf0e9372ef04d3cc10104f1980f54d57b2ce73a5ad56a",
      strip_prefix = "six-1.10.0",
      build_file = str(Label("//third_party:six.BUILD")),
  )

  tf_http_archive(
      name = "absl_py",
      urls = [
          "https://mirror.bazel.build/github.com/abseil/abseil-py/archive/acec853355ef987eae48a8d87a79351c15dff593.tar.gz",
          "https://github.com/abseil/abseil-py/archive/acec853355ef987eae48a8d87a79351c15dff593.tar.gz",
      ],
      sha256 = "29e4584e778bee13aa4093824133d131d927cc160561892880118d9ff7b95a6a",
      strip_prefix = "abseil-py-acec853355ef987eae48a8d87a79351c15dff593",
  )

  tf_http_archive(
      name = "org_python_pypi_backports_weakref",
      urls = [
          "https://mirror.bazel.build/pypi.python.org/packages/bc/cc/3cdb0a02e7e96f6c70bd971bc8a90b8463fda83e264fa9c5c1c98ceabd81/backports.weakref-1.0rc1.tar.gz",
          "https://pypi.python.org/packages/bc/cc/3cdb0a02e7e96f6c70bd971bc8a90b8463fda83e264fa9c5c1c98ceabd81/backports.weakref-1.0rc1.tar.gz",
      ],
      sha256 = "8813bf712a66b3d8b85dc289e1104ed220f1878cf981e2fe756dfaabe9a82892",
      strip_prefix = "backports.weakref-1.0rc1/src",
      build_file = str(Label("//third_party:backports_weakref.BUILD")),
  )

  tf_http_archive(
      name = "com_github_andreif_codegen",
      urls = [
          "https://mirror.bazel.build/github.com/andreif/codegen/archive/1.0.tar.gz",
          "https://github.com/andreif/codegen/archive/1.0.tar.gz",
      ],
      sha256 = "2dadd04a2802de27e0fe5a19b76538f6da9d39ff244036afa00c1bba754de5ee",
      strip_prefix = "codegen-1.0",
      build_file = str(Label("//third_party:codegen.BUILD")),
  )

  filegroup_external(
      name = "org_python_license",
      licenses = ["notice"],  # Python 2.0
      sha256_urls = {
          "b5556e921715ddb9242c076cae3963f483aa47266c5e37ea4c187f77cc79501c": [
              "https://mirror.bazel.build/docs.python.org/2.7/_sources/license.txt",
              "https://docs.python.org/2.7/_sources/license.txt",
          ],
      },
  )

  tf_http_archive(
      name = "protobuf_archive",
      urls = [
          "https://mirror.bazel.build/github.com/google/protobuf/archive/b04e5cba356212e4e8c66c61bbe0c3a20537c5b9.tar.gz",
          "https://github.com/google/protobuf/archive/b04e5cba356212e4e8c66c61bbe0c3a20537c5b9.tar.gz",
      ],
      sha256 = "e178a25c52efcb6b05988bdbeace4c0d3f2d2fe5b46696d1d9898875c3803d6a",
      strip_prefix = "protobuf-b04e5cba356212e4e8c66c61bbe0c3a20537c5b9",
      # TODO: remove patching when tensorflow stops linking same protos into
      #       multiple shared libraries loaded in runtime by python.
      #       This patch fixes a runtime crash when tensorflow is compiled
      #       with clang -O2 on Linux (see https://github.com/tensorflow/tensorflow/issues/8394)
      patch_file = str(Label("//third_party/protobuf:add_noinlines.patch")),
  )

  # We need to import the protobuf library under the names com_google_protobuf
  # and com_google_protobuf_cc to enable proto_library support in bazel.
  # Unfortunately there is no way to alias http_archives at the moment.
  tf_http_archive(
      name = "com_google_protobuf",
      urls = [
          "https://mirror.bazel.build/github.com/google/protobuf/archive/b04e5cba356212e4e8c66c61bbe0c3a20537c5b9.tar.gz",
          "https://github.com/google/protobuf/archive/b04e5cba356212e4e8c66c61bbe0c3a20537c5b9.tar.gz",
      ],
      sha256 = "e178a25c52efcb6b05988bdbeace4c0d3f2d2fe5b46696d1d9898875c3803d6a",
      strip_prefix = "protobuf-b04e5cba356212e4e8c66c61bbe0c3a20537c5b9",
  )

  tf_http_archive(
      name = "com_google_protobuf_cc",
      urls = [
          "https://mirror.bazel.build/github.com/google/protobuf/archive/b04e5cba356212e4e8c66c61bbe0c3a20537c5b9.tar.gz",
          "https://github.com/google/protobuf/archive/b04e5cba356212e4e8c66c61bbe0c3a20537c5b9.tar.gz",
      ],
      sha256 = "e178a25c52efcb6b05988bdbeace4c0d3f2d2fe5b46696d1d9898875c3803d6a",
      strip_prefix = "protobuf-b04e5cba356212e4e8c66c61bbe0c3a20537c5b9",
  )

  tf_http_archive(
      name = "nsync",
      urls = [
          "https://mirror.bazel.build/github.com/google/nsync/archive/8502189abfa44c249c01c2cad64e6ed660a9a668.tar.gz",
          "https://github.com/google/nsync/archive/8502189abfa44c249c01c2cad64e6ed660a9a668.tar.gz",
      ],
      sha256 = "51f81ff4202bbb820cdbedc061bd2eb6765f2b5c06489e7a8694bedac329e8f8",
      strip_prefix = "nsync-8502189abfa44c249c01c2cad64e6ed660a9a668",
  )

  tf_http_archive(
      name = "com_google_googletest",
      urls = [
          "https://mirror.bazel.build/github.com/google/googletest/archive/9816b96a6ddc0430671693df90192bbee57108b6.zip",
          "https://github.com/google/googletest/archive/9816b96a6ddc0430671693df90192bbee57108b6.zip",
      ],
      sha256 = "9cbca84c4256bed17df2c8f4d00c912c19d247c11c9ba6647cd6dd5b5c996b8d",
      strip_prefix = "googletest-9816b96a6ddc0430671693df90192bbee57108b6",
  )

  tf_http_archive(
      name = "com_github_gflags_gflags",
      urls = [
          "https://mirror.bazel.build/github.com/gflags/gflags/archive/f8a0efe03aa69b3336d8e228b37d4ccb17324b88.tar.gz",
          "https://github.com/gflags/gflags/archive/f8a0efe03aa69b3336d8e228b37d4ccb17324b88.tar.gz",
      ],
      sha256 = "4d222fab8f1ede4709cdff417d15a1336f862d7334a81abf76d09c15ecf9acd1",
      strip_prefix = "gflags-f8a0efe03aa69b3336d8e228b37d4ccb17324b88",
  )

  tf_http_archive(
      name = "pcre",
      sha256 = "ccdf7e788769838f8285b3ee672ed573358202305ee361cfec7a4a4fb005bbc7",
      urls = [
          "https://mirror.bazel.build/ftp.exim.org/pub/pcre/pcre-8.39.tar.gz",
          "http://ftp.exim.org/pub/pcre/pcre-8.39.tar.gz",
      ],
      strip_prefix = "pcre-8.39",
      build_file = str(Label("//third_party:pcre.BUILD")),
  )

  tf_http_archive(
      name = "swig",
      sha256 = "58a475dbbd4a4d7075e5fe86d4e54c9edde39847cdb96a3053d87cb64a23a453",
      urls = [
          "https://mirror.bazel.build/ufpr.dl.sourceforge.net/project/swig/swig/swig-3.0.8/swig-3.0.8.tar.gz",
          "http://ufpr.dl.sourceforge.net/project/swig/swig/swig-3.0.8/swig-3.0.8.tar.gz",
          "http://pilotfiber.dl.sourceforge.net/project/swig/swig/swig-3.0.8/swig-3.0.8.tar.gz",
      ],
      strip_prefix = "swig-3.0.8",
      build_file = str(Label("//third_party:swig.BUILD")),
  )

  tf_http_archive(
      name = "curl",
      sha256 = "ff3e80c1ca6a068428726cd7dd19037a47cc538ce58ef61c59587191039b2ca6",
      urls = [
          "https://mirror.bazel.build/curl.haxx.se/download/curl-7.49.1.tar.gz",
          "https://curl.haxx.se/download/curl-7.49.1.tar.gz",
      ],
      strip_prefix = "curl-7.49.1",
      build_file = str(Label("//third_party:curl.BUILD")),
  )

  tf_http_archive(
      name = "grpc",
      urls = [
          "https://mirror.bazel.build/github.com/grpc/grpc/archive/f836c7e941beb003289dc6e9a58a6e47f5caa5f0.tar.gz",
          "https://github.com/grpc/grpc/archive/f836c7e941beb003289dc6e9a58a6e47f5caa5f0.tar.gz",
      ],
      sha256 = "676425fc19e0290443b21f1804e5d1096456b6512b349606e3eae8e63299e6ee",
      strip_prefix = "grpc-f836c7e941beb003289dc6e9a58a6e47f5caa5f0",
  )

  tf_http_archive(
      name = "linenoise",
      sha256 = "7f51f45887a3d31b4ce4fa5965210a5e64637ceac12720cfce7954d6a2e812f7",
      urls = [
          "https://mirror.bazel.build/github.com/antirez/linenoise/archive/c894b9e59f02203dbe4e2be657572cf88c4230c3.tar.gz",
          "https://github.com/antirez/linenoise/archive/c894b9e59f02203dbe4e2be657572cf88c4230c3.tar.gz",
      ],
      strip_prefix = "linenoise-c894b9e59f02203dbe4e2be657572cf88c4230c3",
      build_file = str(Label("//third_party:linenoise.BUILD")),
  )

  # TODO(phawkins): currently, this rule uses an unofficial LLVM mirror.
  # Switch to an official source of snapshots if/when possible.
  tf_http_archive(
      name = "llvm",
      urls = [
          "https://mirror.bazel.build/github.com/llvm-mirror/llvm/archive/9ab4c272cb604a7f947865428c4ef2169fee2100.tar.gz",
          "https://github.com/llvm-mirror/llvm/archive/9ab4c272cb604a7f947865428c4ef2169fee2100.tar.gz",
      ],
      sha256 = "1b1b7d3800a94ca2302e3dd670dbe84238749583027883784b55297059d83da8",
      strip_prefix = "llvm-9ab4c272cb604a7f947865428c4ef2169fee2100",
      build_file = str(Label("//third_party/llvm:llvm.BUILD")),
  )

  tf_http_archive(
      name = "lmdb",
      urls = [
          "https://mirror.bazel.build/github.com/LMDB/lmdb/archive/LMDB_0.9.19.tar.gz",
          "https://github.com/LMDB/lmdb/archive/LMDB_0.9.19.tar.gz",
      ],
      sha256 = "108532fb94c6f227558d45be3f3347b52539f0f58290a7bb31ec06c462d05326",
      strip_prefix = "lmdb-LMDB_0.9.19/libraries/liblmdb",
      build_file = str(Label("//third_party:lmdb.BUILD")),
  )

  tf_http_archive(
      name = "jsoncpp_git",
      urls = [
          "https://mirror.bazel.build/github.com/open-source-parsers/jsoncpp/archive/11086dd6a7eba04289944367ca82cea71299ed70.tar.gz",
          "https://github.com/open-source-parsers/jsoncpp/archive/11086dd6a7eba04289944367ca82cea71299ed70.tar.gz",
      ],
      sha256 = "07d34db40593d257324ec5fb9debc4dc33f29f8fb44e33a2eeb35503e61d0fe2",
      strip_prefix = "jsoncpp-11086dd6a7eba04289944367ca82cea71299ed70",
      build_file = str(Label("//third_party:jsoncpp.BUILD")),
  )

  tf_http_archive(
      name = "boringssl",
      urls = [
          "https://mirror.bazel.build/github.com/google/boringssl/archive/a0fb951d2a26a8ee746b52f3ba81ab011a0af778.tar.gz",
          "https://github.com/google/boringssl/archive/a0fb951d2a26a8ee746b52f3ba81ab011a0af778.tar.gz",
      ],
      sha256 = "524ba98a56300149696481b4cb9ddebd0c7b7ac9b9f6edee81da2d2d7e5d2bb3",
      strip_prefix = "boringssl-a0fb951d2a26a8ee746b52f3ba81ab011a0af778",
  )

  tf_http_archive(
      name = "zlib_archive",
      urls = [
          "https://mirror.bazel.build/zlib.net/zlib-1.2.8.tar.gz",
          "http://zlib.net/fossils/zlib-1.2.8.tar.gz",
      ],
      sha256 = "36658cb768a54c1d4dec43c3116c27ed893e88b02ecfcb44f2166f9c0b7f2a0d",
      strip_prefix = "zlib-1.2.8",
      build_file = str(Label("//third_party:zlib.BUILD")),
  )

  tf_http_archive(
      name = "fft2d",
      urls = [
          "https://mirror.bazel.build/www.kurims.kyoto-u.ac.jp/~ooura/fft.tgz",
          "http://www.kurims.kyoto-u.ac.jp/~ooura/fft.tgz",
      ],
      sha256 = "52bb637c70b971958ec79c9c8752b1df5ff0218a4db4510e60826e0cb79b5296",
      build_file = str(Label("//third_party/fft2d:fft2d.BUILD")),
  )

  tf_http_archive(
      name = "snappy",
      urls = [
          "https://mirror.bazel.build/github.com/google/snappy/archive/1.1.4.tar.gz",
          "https://github.com/google/snappy/archive/1.1.4.tar.gz",
      ],
      sha256 = "2f7504c73d85bac842e893340333be8cb8561710642fc9562fccdd9d2c3fcc94",
      strip_prefix = "snappy-1.1.4",
      build_file = str(Label("//third_party:snappy.BUILD")),
  )

  tf_http_archive(
      name = "nccl_archive",
      urls = [
          "https://mirror.bazel.build/github.com/nvidia/nccl/archive/03d856977ecbaac87e598c0c4bafca96761b9ac7.tar.gz",
          "https://github.com/nvidia/nccl/archive/03d856977ecbaac87e598c0c4bafca96761b9ac7.tar.gz",
      ],
      sha256 = "2ca86fb6179ecbff789cc67c836139c1bbc0324ed8c04643405a30bf26325176",
      strip_prefix = "nccl-03d856977ecbaac87e598c0c4bafca96761b9ac7",
      build_file = str(Label("//third_party:nccl.BUILD")),
  )

  tf_http_archive(
      name = "aws",
      urls = [
          "https://mirror.bazel.build/github.com/aws/aws-sdk-cpp/archive/1.3.15.tar.gz",
          "https://github.com/aws/aws-sdk-cpp/archive/1.3.15.tar.gz",
      ],
      sha256 = "b888d8ce5fc10254c3dd6c9020c7764dd53cf39cf011249d0b4deda895de1b7c",
      strip_prefix = "aws-sdk-cpp-1.3.15",
      build_file = str(Label("//third_party:aws.BUILD")),
  )

  java_import_external(
      name = "junit",
      jar_sha256 = "59721f0805e223d84b90677887d9ff567dc534d7c502ca903c0c2b17f05c116a",
      jar_urls = [
          "https://mirror.bazel.build/repo1.maven.org/maven2/junit/junit/4.12/junit-4.12.jar",
          "http://repo1.maven.org/maven2/junit/junit/4.12/junit-4.12.jar",
          "http://maven.ibiblio.org/maven2/junit/junit/4.12/junit-4.12.jar",
      ],
      licenses = ["reciprocal"],  # Common Public License Version 1.0
      testonly_ = True,
      deps = ["@org_hamcrest_core"],
  )

  java_import_external(
      name = "org_hamcrest_core",
      jar_sha256 = "66fdef91e9739348df7a096aa384a5685f4e875584cce89386a7a47251c4d8e9",
      jar_urls = [
          "https://mirror.bazel.build/repo1.maven.org/maven2/org/hamcrest/hamcrest-core/1.3/hamcrest-core-1.3.jar",
          "http://repo1.maven.org/maven2/org/hamcrest/hamcrest-core/1.3/hamcrest-core-1.3.jar",
          "http://maven.ibiblio.org/maven2/org/hamcrest/hamcrest-core/1.3/hamcrest-core-1.3.jar",
      ],
      licenses = ["notice"],  # New BSD License
      testonly_ = True,
  )

  tf_http_archive(
      name = "jemalloc",
      urls = [
          "https://mirror.bazel.build/github.com/jemalloc/jemalloc/archive/4.4.0.tar.gz",
          "https://github.com/jemalloc/jemalloc/archive/4.4.0.tar.gz",
      ],
      sha256 = "3c8f25c02e806c3ce0ab5fb7da1817f89fc9732709024e2a81b6b82f7cc792a8",
      strip_prefix = "jemalloc-4.4.0",
      build_file = str(Label("//third_party:jemalloc.BUILD")),
  )

  java_import_external(
      name = "com_google_testing_compile",
      jar_sha256 = "edc180fdcd9f740240da1a7a45673f46f59c5578d8cd3fbc912161f74b5aebb8",
      jar_urls = [
          "http://mirror.bazel.build/repo1.maven.org/maven2/com/google/testing/compile/compile-testing/0.11/compile-testing-0.11.jar",
          "http://repo1.maven.org/maven2/com/google/testing/compile/compile-testing/0.11/compile-testing-0.11.jar",
          "http://maven.ibiblio.org/maven2/com/google/testing/compile/compile-testing/0.11/compile-testing-0.11.jar",
      ],
      licenses = ["notice"],  # New BSD License
      testonly_ = True,
      deps = ["@com_google_guava", "@com_google_truth"],
  )

  java_import_external(
      name = "com_google_truth",
      jar_sha256 = "032eddc69652b0a1f8d458f999b4a9534965c646b8b5de0eba48ee69407051df",
      jar_urls = [
          "http://mirror.bazel.build/repo1.maven.org/maven2/com/google/truth/truth/0.32/truth-0.32.jar",
          "http://repo1.maven.org/maven2/com/google/truth/truth/0.32/truth-0.32.jar",
      ],
      licenses = ["notice"],  # Apache 2.0
      testonly_ = True,
      deps = ["@com_google_guava"],
  )

  java_import_external(
      name = "javax_validation",
      jar_sha256 = "e459f313ebc6db2483f8ceaad39af07086361b474fa92e40f442e8de5d9895dc",
      jar_urls = [
          "http://mirror.bazel.build/repo1.maven.org/maven2/javax/validation/validation-api/1.0.0.GA/validation-api-1.0.0.GA.jar",
          "http://repo1.maven.org/maven2/javax/validation/validation-api/1.0.0.GA/validation-api-1.0.0.GA.jar",
      ],
      licenses = ["notice"],  # Apache 2.0
  )

  tf_http_archive(
      name = "com_google_pprof",
      urls = [
          "https://mirror.bazel.build/github.com/google/pprof/archive/c0fb62ec88c411cc91194465e54db2632845b650.tar.gz",
          "https://github.com/google/pprof/archive/c0fb62ec88c411cc91194465e54db2632845b650.tar.gz",
      ],
      sha256 = "e0928ca4aa10ea1e0551e2d7ce4d1d7ea2d84b2abbdef082b0da84268791d0c4",
      strip_prefix = "pprof-c0fb62ec88c411cc91194465e54db2632845b650",
      build_file = str(Label("//third_party:pprof.BUILD")),
  )

  tf_http_archive(
      name = "cub_archive",
      urls = [
          "https://mirror.bazel.build/github.com/NVlabs/cub/archive/1.7.4.zip",
          "https://github.com/NVlabs/cub/archive/1.7.4.zip",
      ],
      sha256 = "20a1a39fd97e5da7f40f5f2e7fd73fd2ea59f9dc4bb8a6c5f228aa543e727e31",
      strip_prefix = "cub-1.7.4",
      build_file = str(Label("//third_party:cub.BUILD")),
  )

  tf_http_archive(
      name = "cython",
      sha256 = "6dcd30b5ceb887b2b965ee7ceb82ea3acb5f0642fe2206c7636b45acea4798e5",
      urls = [
          "https://mirror.bazel.build/github.com/cython/cython/archive/3732784c45cfb040a5b0936951d196f83a12ea17.tar.gz",
          "https://github.com/cython/cython/archive/3732784c45cfb040a5b0936951d196f83a12ea17.tar.gz",
      ],
      strip_prefix = "cython-3732784c45cfb040a5b0936951d196f83a12ea17",
      build_file = str(Label("//third_party:cython.BUILD")),
      delete = ["BUILD.bazel"],
  )

  tf_http_archive(
      name = "bazel_toolchains",
      urls = [
          "https://mirror.bazel.build/github.com/bazelbuild/bazel-toolchains/archive/b49ba3689f46ac50e9277dafd8ff32b26951f82e.tar.gz",
          "https://github.com/bazelbuild/bazel-toolchains/archive/b49ba3689f46ac50e9277dafd8ff32b26951f82e.tar.gz",
      ],
      sha256 = "1266f1e27b4363c83222f1a776397c7a069fbfd6aacc9559afa61cdd73e1b429",
      strip_prefix = "bazel-toolchains-b49ba3689f46ac50e9277dafd8ff32b26951f82e",
  )

  tf_http_archive(
      name = "arm_neon_2_x86_sse",
      sha256 = "c8d90aa4357f8079d427e87a6f4c493da1fa4140aee926c05902d7ec1533d9a5",
      strip_prefix = "ARM_NEON_2_x86_SSE-0f77d9d182265259b135dad949230ecbf1a2633d",
      urls = [
          "https://mirror.bazel.build/github.com/intel/ARM_NEON_2_x86_SSE/archive/0f77d9d182265259b135dad949230ecbf1a2633d.tar.gz",
          "https://github.com/intel/ARM_NEON_2_x86_SSE/archive/0f77d9d182265259b135dad949230ecbf1a2633d.tar.gz",
      ],
      build_file = str(Label("//third_party:arm_neon_2_x86_sse.BUILD")),
  )

  tf_http_archive(
      name = "flatbuffers",
      strip_prefix = "flatbuffers-971a68110e4fc1bace10fcb6deeb189e7e1a34ce",
      sha256 = "874088d2ee0d9f8524191f77209556415f03dd44e156276edf19e5b90ceb5f55",
      urls = [
          "https://mirror.bazel.build/github.com/google/flatbuffers/archive/971a68110e4fc1bace10fcb6deeb189e7e1a34ce.tar.gz",
          "https://github.com/google/flatbuffers/archive/971a68110e4fc1bace10fcb6deeb189e7e1a34ce.tar.gz",
      ],
      build_file = str(Label("//third_party/flatbuffers:flatbuffers.BUILD")),
  )

  tf_http_archive(
      name = "tflite_mobilenet",
      sha256 = "23f814d1c076bdf03715dfb6cab3713aa4fbdf040fd5448c43196bd2e97a4c1b",
      urls = [
          "https://mirror.bazel.build/storage.googleapis.com/download.tensorflow.org/models/tflite/mobilenet_v1_224_android_quant_2017_11_08.zip",
          "https://storage.googleapis.com/download.tensorflow.org/models/tflite/mobilenet_v1_224_android_quant_2017_11_08.zip",
      ],
      build_file = str(Label("//third_party:tflite_mobilenet.BUILD")),
  )

  tf_http_archive(
      name = "tflite_smartreply",
      sha256 = "8980151b85a87a9c1a3bb1ed4748119e4a85abd3cb5744d83da4d4bd0fbeef7c",
      urls = [
          "https://mirror.bazel.build/storage.googleapis.com/download.tensorflow.org/models/tflite/smartreply_1.0_2017_11_01.zip",
          "https://storage.googleapis.com/download.tensorflow.org/models/tflite/smartreply_1.0_2017_11_01.zip"
      ],
      build_file = str(Label("//third_party:tflite_smartreply.BUILD")),
  )

  ##############################################################################
  # BIND DEFINITIONS
  #
  # Please do not add bind() definitions unless we have no other choice.
  # If that ends up being the case, please leave a comment explaining
  # why we can't depend on the canonical build target.

  # gRPC wants a cares dependency but its contents is not actually
  # important since we have set GRPC_ARES=0 in tools/bazel.rc
  native.bind(
      name = "cares",
      actual = "@grpc//third_party/nanopb:nanopb",
  )

  # Needed by Protobuf
  native.bind(
      name = "grpc_cpp_plugin",
      actual = "@grpc//:grpc_cpp_plugin",
  )

  # gRPC has three empty C++ functions which it wants the user to define
  # at build time. https://github.com/grpc/grpc/issues/13590
  native.bind(
      name = "grpc_lib",
      actual = "@grpc//:grpc++_unsecure",
  )

  # Needed by gRPC
  native.bind(
      name = "libssl",
      actual = "@boringssl//:ssl",
  )

  # Needed by gRPC
  native.bind(
      name = "nanopb",
      actual = "@grpc//third_party/nanopb:nanopb",
  )

  # Needed by gRPC
  native.bind(
      name = "protobuf",
      actual = "@protobuf_archive//:protobuf",
  )

  # gRPC expects //external:protobuf_clib and //external:protobuf_compiler
  # to point to Protobuf's compiler library.
  native.bind(
      name = "protobuf_clib",
      actual = "@protobuf_archive//:protoc_lib",
  )

  # Needed by gRPC
  native.bind(
      name = "protobuf_headers",
      actual = "@protobuf_archive//:protobuf_headers",
  )

  # Needed by Protobuf
  native.bind(
      name = "python_headers",
      actual = str(Label("//util/python:python_headers")),
  )

  # Needed by Protobuf
  native.bind(
      name = "six",
      actual = "@six_archive//:six",
  )

  # Needed by gRPC
  native.bind(
      name = "zlib",
      actual = "@zlib_archive//:zlib",
  )
