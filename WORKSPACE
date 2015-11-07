<<<<<<< HEAD
workspace(name = "org_tensorflow")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive", "http_file")

http_archive(
    name = "io_bazel_rules_closure",
    sha256 = "5b00383d08dd71f28503736db0500b6fb4dda47489ff5fc6bed42557c07c6ba9",
    strip_prefix = "rules_closure-308b05b2419edb5c8ee0471b67a40403df940149",
    urls = [
        "https://storage.googleapis.com/mirror.tensorflow.org/github.com/bazelbuild/rules_closure/archive/308b05b2419edb5c8ee0471b67a40403df940149.tar.gz",
        "https://github.com/bazelbuild/rules_closure/archive/308b05b2419edb5c8ee0471b67a40403df940149.tar.gz",  # 2019-06-13
    ],
)

# Load tf_repositories() before loading dependencies for other repository so
# that dependencies like com_google_protobuf won't be overridden.
load("//tensorflow:workspace.bzl", "tf_repositories")
# Please add all new TensorFlow dependencies in workspace.bzl.
tf_repositories()

register_toolchains("@local_config_python//:py_toolchain")

load("@io_bazel_rules_closure//closure:defs.bzl", "closure_repositories")

closure_repositories()

load("//third_party/toolchains/preconfig/generate:archives.bzl",
     "bazel_toolchains_archive")

bazel_toolchains_archive()

load(
    "@bazel_toolchains//repositories:repositories.bzl",
    bazel_toolchains_repositories = "repositories",
)

bazel_toolchains_repositories()

load(
    "@io_bazel_rules_docker//repositories:repositories.bzl",
    container_repositories = "repositories",
)

container_repositories()

load("//third_party/toolchains/preconfig/generate:workspace.bzl",
     "remote_config_workspace")

remote_config_workspace()

# Apple and Swift rules.
http_archive(
    name = "build_bazel_rules_apple",
    sha256 = "a045a436b642c70fb0c10ca84ff0fd2dcbd59cc89100d597a61e8374afafb366",
    urls = ["https://github.com/bazelbuild/rules_apple/releases/download/0.18.0/rules_apple.0.18.0.tar.gz"],
)  # https://github.com/bazelbuild/rules_apple/releases
http_archive(
    name = "build_bazel_rules_swift",
    sha256 = "18cd4df4e410b0439a4935f9ca035bd979993d42372ba79e7f2d4fafe9596ef0",
    urls = ["https://github.com/bazelbuild/rules_swift/releases/download/0.12.1/rules_swift.0.12.1.tar.gz"],
)  # https://github.com/bazelbuild/rules_swift/releases
http_archive(
    name = "build_bazel_apple_support",
    sha256 = "122ebf7fe7d1c8e938af6aeaee0efe788a3a2449ece5a8d6a428cb18d6f88033",
    urls = ["https://github.com/bazelbuild/apple_support/releases/download/0.7.1/apple_support.0.7.1.tar.gz"],
)  # https://github.com/bazelbuild/apple_support/releases
http_archive(
    name = "bazel_skylib",
    sha256 = "1dde365491125a3db70731e25658dfdd3bc5dbdfd11b840b3e987ecf043c7ca0",
    urls = ["https://github.com/bazelbuild/bazel-skylib/releases/download/0.9.0/bazel-skylib.0.9.0.tar.gz"],
)  # https://github.com/bazelbuild/bazel-skylib/releases
http_archive(
    name = "com_github_apple_swift_swift_protobuf",
    type = "zip",
    strip_prefix = "swift-protobuf-1.6.0/",
    urls = ["https://github.com/apple/swift-protobuf/archive/1.6.0.zip"],
)  # https://github.com/apple/swift-protobuf/releases
http_file(
    name = "xctestrunner",
    executable = 1,
    urls = ["https://github.com/google/xctestrunner/releases/download/0.2.9/ios_test_runner.par"],
)  # https://github.com/google/xctestrunner/releases
# Use `swift_rules_dependencies` to fetch the toolchains. With the
# `git_repository` rules above, the following call will skip redefining them.
load("@build_bazel_rules_swift//swift:repositories.bzl", "swift_rules_dependencies")
swift_rules_dependencies()

# We must check the bazel version before trying to parse any other BUILD
# files, in case the parsing of those build files depends on the bazel
# version we require here.
load("//tensorflow:version_check.bzl", "check_bazel_version_at_least")
check_bazel_version_at_least("1.0.0")

load("//third_party/android:android_configure.bzl", "android_configure")
android_configure(name="local_config_android")
load("@local_config_android//:android.bzl", "android_workspace")
android_workspace()

# If a target is bound twice, the later one wins, so we have to do tf bindings
# at the end of the WORKSPACE file.
load("//tensorflow:workspace.bzl", "tf_bind")
tf_bind()

http_archive(
    name = "inception_v1",
    build_file = "//:models.BUILD",
    sha256 = "7efe12a8363f09bc24d7b7a450304a15655a57a7751929b2c1593a71183bb105",
    urls = [
        "https://storage.googleapis.com/download.tensorflow.org/models/inception_v1.zip",
    ],
)

http_archive(
    name = "mobile_ssd",
    build_file = "//:models.BUILD",
    sha256 = "bddd81ea5c80a97adfac1c9f770e6f55cbafd7cce4d3bbe15fbeb041e6b8f3e8",
    urls = [
        "https://storage.googleapis.com/download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_android_export.zip",
    ],
)

http_archive(
    name = "mobile_multibox",
    build_file = "//:models.BUILD",
    sha256 = "859edcddf84dddb974c36c36cfc1f74555148e9c9213dedacf1d6b613ad52b96",
    urls = [
        "https://storage.googleapis.com/download.tensorflow.org/models/mobile_multibox_v1a.zip",
    ],
)

http_archive(
    name = "stylize",
    build_file = "//:models.BUILD",
    sha256 = "3d374a730aef330424a356a8d4f04d8a54277c425e274ecb7d9c83aa912c6bfa",
    urls = [
        "https://storage.googleapis.com/download.tensorflow.org/models/stylize_v1.zip",
    ],
)

http_archive(
    name = "speech_commands",
    build_file = "//:models.BUILD",
    sha256 = "c3ec4fea3158eb111f1d932336351edfe8bd515bb6e87aad4f25dbad0a600d0c",
    urls = [
        "https://storage.googleapis.com/download.tensorflow.org/models/speech_commands_v0.01.zip",
    ],
=======
# Uncomment and update the paths in these entries to build the Android demo.
#android_sdk_repository(
#    name = "androidsdk",
#    api_level = 23,
#    build_tools_version = "23.0.1",
#    # Replace with path to Android SDK on your system
#    path = "<PATH_TO_SDK>",
#)
#
#android_ndk_repository(
#    name="androidndk",
#    path="<PATH_TO_NDK>",
#    api_level=21)

new_http_archive(
  name = "gmock_archive",
  url = "https://googlemock.googlecode.com/files/gmock-1.7.0.zip",
  sha256 = "26fcbb5925b74ad5fc8c26b0495dfc96353f4d553492eb97e85a8a6d2f43095b",
  build_file = "gmock.BUILD",
)

bind(
  name = "gtest",
  actual = "@gmock_archive//:gtest",
)

bind(
  name = "gtest_main",
  actual = "@gmock_archive//:gtest_main",
)

git_repository(
  name = "re2",
  remote = "https://github.com/google/re2.git",
  tag = "2015-07-01",
)

new_http_archive(
  name = "jpeg_archive",
  url = "http://www.ijg.org/files/jpegsrc.v9a.tar.gz",
  sha256 = "3a753ea48d917945dd54a2d97de388aa06ca2eb1066cbfdc6652036349fe05a7",
  build_file = "jpeg.BUILD",
)

git_repository(
  name = "gemmlowp",
  remote = "https://github.com/google/gemmlowp.git",
  commit = "cc5d3a0",
)

new_http_archive(
  name = "png_archive",
  url = "https://storage.googleapis.com/libpng-public-archive/libpng-1.2.53.tar.gz",
  sha256 = "e05c9056d7f323088fd7824d8c6acc03a4a758c4b4916715924edc5dd3223a72",
  build_file = "png.BUILD",
)

new_http_archive(
  name = "six_archive",
  url = "https://pypi.python.org/packages/source/s/six/six-1.10.0.tar.gz#md5=34eed507548117b2ab523ab14b2f8b55",
  sha256 = "105f8d68616f8248e24bf0e9372ef04d3cc10104f1980f54d57b2ce73a5ad56a",
  build_file = "six.BUILD",
)

bind(
  name = "six",
  actual = "@six_archive//:six",
)

new_git_repository(
  name = "iron-ajax",
  build_file = "bower.BUILD",
  remote = "https://github.com/PolymerElements/iron-ajax.git",
  tag = "v1.0.8",
)

new_git_repository(
  name = "iron-dropdown",
  build_file = "bower.BUILD",
  remote = "https://github.com/PolymerElements/iron-dropdown.git",
  tag = "v1.0.6",
)

new_git_repository(
  name = "accessibility-developer-tools",
  build_file = "bower.BUILD",
  remote = "https://github.com/GoogleChrome/accessibility-developer-tools.git",
  tag = "v2.10.0",
)

new_git_repository(
  name = "iron-doc-viewer",
  build_file = "bower.BUILD",
  remote = "https://github.com/PolymerElements/iron-doc-viewer.git",
  tag = "v1.0.6",
)

new_git_repository(
  name = "iron-icons",
  build_file = "bower.BUILD",
  remote = "https://github.com/polymerelements/iron-icons.git",
  tag = "v1.0.4",
)

new_git_repository(
  name = "paper-icon-button",
  build_file = "bower.BUILD",
  remote = "https://github.com/PolymerElements/paper-icon-button.git",
  tag = "v1.0.5",
)

new_git_repository(
  name = "sinonjs",
  build_file = "bower.BUILD",
  remote = "https://github.com/blittle/sinon.js.git",
  tag = "v1.17.1",
)

new_git_repository(
  name = "paper-dropdown-menu",
  build_file = "bower.BUILD",
  remote = "https://github.com/PolymerElements/paper-dropdown-menu.git",
  tag = "v1.0.5",
)

new_git_repository(
  name = "iron-flex-layout",
  build_file = "bower.BUILD",
  remote = "https://github.com/polymerelements/iron-flex-layout.git",
  tag = "v1.0.4",
)

new_git_repository(
  name = "iron-autogrow-textarea",
  build_file = "bower.BUILD",
  remote = "https://github.com/PolymerElements/iron-autogrow-textarea.git",
  tag = "v1.0.7",
)

new_git_repository(
  name = "d3",
  build_file = "bower.BUILD",
  remote = "https://github.com/mbostock/d3.git",
  tag = "v3.5.6",
)

new_git_repository(
  name = "iron-component-page",
  build_file = "bower.BUILD",
  remote = "https://github.com/PolymerElements/iron-component-page.git",
  tag = "v1.0.8",
)

new_git_repository(
  name = "stacky",
  build_file = "bower.BUILD",
  remote = "https://github.com/PolymerLabs/stacky.git",
  tag = "v1.2.4",
)

new_git_repository(
  name = "paper-styles",
  build_file = "bower.BUILD",
  remote = "https://github.com/PolymerElements/paper-styles.git",
  tag = "v1.0.12",
)

new_git_repository(
  name = "paper-input",
  build_file = "bower.BUILD",
  remote = "https://github.com/PolymerElements/paper-input.git",
  tag = "v1.0.16",
)

new_git_repository(
  name = "paper-item",
  build_file = "bower.BUILD",
  remote = "https://github.com/PolymerElements/paper-item.git",
  tag = "v1.0.5",
)

new_git_repository(
  name = "marked-element",
  build_file = "bower.BUILD",
  remote = "https://github.com/PolymerElements/marked-element.git",
  tag = "v1.1.1",
)

new_git_repository(
  name = "prism",
  build_file = "bower.BUILD",
  remote = "https://github.com/LeaVerou/prism.git",
  tag = "v1.3.0",
)

new_git_repository(
  name = "paper-progress",
  build_file = "bower.BUILD",
  remote = "https://github.com/PolymerElements/paper-progress.git",
  tag = "v1.0.7",
)

new_git_repository(
  name = "iron-checked-element-behavior",
  build_file = "bower.BUILD",
  remote = "https://github.com/PolymerElements/iron-checked-element-behavior.git",
  tag = "v1.0.2",
)

new_git_repository(
  name = "paper-toolbar",
  build_file = "bower.BUILD",
  remote = "https://github.com/PolymerElements/paper-toolbar.git",
  tag = "v1.0.4",
)

new_git_repository(
  name = "async",
  build_file = "bower.BUILD",
  remote = "https://github.com/caolan/async.git",
  tag = "0.9.2",
)

new_git_repository(
  name = "es6-promise",
  build_file = "bower.BUILD",
  remote = "https://github.com/components/es6-promise.git",
  tag = "v3.0.2",
)

new_git_repository(
  name = "promise-polyfill",
  build_file = "bower.BUILD",
  remote = "https://github.com/polymerlabs/promise-polyfill.git",
  tag = "v1.0.0",
)

new_git_repository(
  name = "font-roboto",
  build_file = "bower.BUILD",
  remote = "https://github.com/PolymerElements/font-roboto.git",
  tag = "v1.0.1",
)

new_git_repository(
  name = "paper-menu",
  build_file = "bower.BUILD",
  remote = "https://github.com/PolymerElements/paper-menu.git",
  tag = "v1.1.1",
)

new_git_repository(
  name = "iron-icon",
  build_file = "bower.BUILD",
  remote = "https://github.com/polymerelements/iron-icon.git",
  tag = "v1.0.7",
)

new_git_repository(
  name = "iron-meta",
  build_file = "bower.BUILD",
  remote = "https://github.com/PolymerElements/iron-meta.git",
  tag = "v1.1.0",
)

new_git_repository(
  name = "lodash",
  build_file = "bower.BUILD",
  remote = "https://github.com/lodash/lodash.git",
  tag = "3.10.1",
)

new_git_repository(
  name = "iron-resizable-behavior",
  build_file = "bower.BUILD",
  remote = "https://github.com/PolymerElements/iron-resizable-behavior.git",
  tag = "v1.0.2",
)

new_git_repository(
  name = "iron-fit-behavior",
  build_file = "bower.BUILD",
  remote = "https://github.com/PolymerElements/iron-fit-behavior.git",
  tag = "v1.0.3",
)

new_git_repository(
  name = "iron-overlay-behavior",
  build_file = "bower.BUILD",
  remote = "https://github.com/PolymerElements/iron-overlay-behavior.git",
  tag = "v1.0.9",
)

new_git_repository(
  name = "neon-animation",
  build_file = "bower.BUILD",
  remote = "https://github.com/polymerelements/neon-animation.git",
  tag = "v1.0.7",
)

new_git_repository(
  name = "iron-a11y-keys-behavior",
  build_file = "bower.BUILD",
  remote = "https://github.com/polymerelements/iron-a11y-keys-behavior.git",
  tag = "v1.0.7",
)

new_git_repository(
  name = "plottable",
  build_file = "bower.BUILD",
  remote = "https://github.com/palantir/plottable.git",
  tag = "v1.16.1",
)

new_git_repository(
  name = "webcomponentsjs",
  build_file = "bower.BUILD",
  remote = "https://github.com/Polymer/webcomponentsjs.git",
  tag = "v0.7.15",
)

new_git_repository(
  name = "iron-validatable-behavior",
  build_file = "bower.BUILD",
  remote = "https://github.com/PolymerElements/iron-validatable-behavior.git",
  tag = "v1.0.5",
)

new_git_repository(
  name = "sinon-chai",
  build_file = "bower.BUILD",
  remote = "https://github.com/domenic/sinon-chai.git",
  tag = "2.8.0",
)

new_git_repository(
  name = "paper-button",
  build_file = "bower.BUILD",
  remote = "https://github.com/PolymerElements/paper-button.git",
  tag = "v1.0.8",
)

new_git_repository(
  name = "iron-input",
  build_file = "bower.BUILD",
  remote = "https://github.com/PolymerElements/iron-input.git",
  tag = "v1.0.6",
)

new_git_repository(
  name = "iron-menu-behavior",
  build_file = "bower.BUILD",
  remote = "https://github.com/PolymerElements/iron-menu-behavior.git",
  tag = "v1.0.5",
)

new_git_repository(
  name = "paper-slider",
  build_file = "bower.BUILD",
  remote = "https://github.com/PolymerElements/paper-slider.git",
  tag = "v1.0.7",
)

new_git_repository(
  name = "iron-list",
  build_file = "bower.BUILD",
  remote = "https://github.com/PolymerElements/iron-list.git",
  tag = "v1.1.5",
)

new_git_repository(
  name = "marked",
  build_file = "bower.BUILD",
  remote = "https://github.com/chjj/marked.git",
  tag = "v0.3.5",
)

new_git_repository(
  name = "paper-material",
  build_file = "bower.BUILD",
  remote = "https://github.com/polymerelements/paper-material.git",
  tag = "v1.0.3",
)

new_git_repository(
  name = "iron-range-behavior",
  build_file = "bower.BUILD",
  remote = "https://github.com/PolymerElements/iron-range-behavior.git",
  tag = "v1.0.4",
)

new_git_repository(
  name = "svg-typewriter",
  build_file = "bower.BUILD",
  remote = "https://github.com/palantir/svg-typewriter.git",
  tag = "v0.3.0",
)

new_git_repository(
  name = "web-animations-js",
  build_file = "bower.BUILD",
  remote = "https://github.com/web-animations/web-animations-js.git",
  tag = "2.1.2",
)

new_git_repository(
  name = "hydrolysis",
  build_file = "bower.BUILD",
  remote = "https://github.com/Polymer/hydrolysis.git",
  tag = "v1.19.3",
)

new_git_repository(
  name = "web-component-tester",
  build_file = "bower.BUILD",
  remote = "https://github.com/Polymer/web-component-tester.git",
  tag = "v3.3.29",
)

new_git_repository(
  name = "paper-toggle-button",
  build_file = "bower.BUILD",
  remote = "https://github.com/PolymerElements/paper-toggle-button.git",
  tag = "v1.0.11",
)

new_git_repository(
  name = "paper-behaviors",
  build_file = "bower.BUILD",
  remote = "https://github.com/polymerelements/paper-behaviors.git",
  tag = "v1.0.7",
)

new_git_repository(
  name = "paper-radio-group",
  build_file = "bower.BUILD",
  remote = "https://github.com/PolymerElements/paper-radio-group.git",
  tag = "v1.0.6",
)

new_git_repository(
  name = "iron-selector",
  build_file = "bower.BUILD",
  remote = "https://github.com/PolymerElements/iron-selector.git",
  tag = "v1.0.7",
)

new_git_repository(
  name = "iron-form-element-behavior",
  build_file = "bower.BUILD",
  remote = "https://github.com/PolymerElements/iron-form-element-behavior.git",
  tag = "v1.0.5",
)

new_git_repository(
  name = "mocha",
  build_file = "bower.BUILD",
  remote = "https://github.com/mochajs/mocha.git",
  tag = "v2.3.3",
)

new_git_repository(
  name = "dagre",
  build_file = "bower.BUILD",
  remote = "https://github.com/cpettitt/dagre.git",
  tag = "v0.7.4",
)

new_git_repository(
  name = "iron-behaviors",
  build_file = "bower.BUILD",
  remote = "https://github.com/PolymerElements/iron-behaviors.git",
  tag = "v1.0.9",
)

new_git_repository(
  name = "graphlib",
  build_file = "bower.BUILD",
  remote = "https://github.com/cpettitt/graphlib.git",
  tag = "v1.0.7",
)

new_git_repository(
  name = "iron-collapse",
  build_file = "bower.BUILD",
  remote = "https://github.com/PolymerElements/iron-collapse.git",
  tag = "v1.0.4",
)

new_git_repository(
  name = "paper-checkbox",
  build_file = "bower.BUILD",
  remote = "https://github.com/PolymerElements/paper-checkbox.git",
  tag = "v1.0.13",
)

new_git_repository(
  name = "paper-radio-button",
  build_file = "bower.BUILD",
  remote = "https://github.com/PolymerElements/paper-radio-button.git",
  tag = "v1.0.10",
)

new_git_repository(
  name = "paper-header-panel",
  build_file = "bower.BUILD",
  remote = "https://github.com/PolymerElements/paper-header-panel.git",
  tag = "v1.0.5",
)

new_git_repository(
  name = "prism-element",
  build_file = "bower.BUILD",
  remote = "https://github.com/PolymerElements/prism-element.git",
  tag = "v1.0.2",
)

new_git_repository(
  name = "chai",
  build_file = "bower.BUILD",
  remote = "https://github.com/chaijs/chai.git",
  tag = "2.3.0",
)

new_git_repository(
  name = "paper-menu-button",
  build_file = "bower.BUILD",
  remote = "https://github.com/polymerelements/paper-menu-button.git",
  tag = "v1.0.3",
)

new_git_repository(
  name = "polymer",
  build_file = "bower.BUILD",
  remote = "https://github.com/Polymer/polymer.git",
  tag = "v1.2.1",
)

new_git_repository(
  name = "paper-ripple",
  build_file = "bower.BUILD",
  remote = "https://github.com/polymerelements/paper-ripple.git",
  tag = "v1.0.4",
)

new_git_repository(
  name = "iron-iconset-svg",
  build_file = "bower.BUILD",
  remote = "https://github.com/PolymerElements/iron-iconset-svg.git",
  tag = "v1.0.8",
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
)
