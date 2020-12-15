# Codes for general utility functions.

load("//devtools/python/blaze:pytype.bzl", "pytype_strict_library")

package(default_visibility = [
    "//vr/perception:__subpackages__",
])

pytype_strict_library(
    name = "network_utils",
    srcs = ["network_utils.py"],
    srcs_version = "PY3",
    deps = [
        "//third_party/py/tensorflow",
        "//third_party/py/tensorflow_addons",
    ],
)

py_test(
    name = "network_utils_test",
    srcs = ["network_utils_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":network_utils",
        "//third_party/py/absl/testing:parameterized",
        "//third_party/py/tensorflow",
        "//third_party/py/tensorflow_addons",
    ],
)

pytype_strict_library(
    name = "network_imnet",
    srcs = ["network_imnet.py"],
    srcs_version = "PY3",
    deps = [
        ":network_utils",
        "//third_party/py/tensorflow",
    ],
)

py_test(
    name = "network_imnet_test",
    srcs = ["network_imnet_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":network_imnet",
        "//third_party/py/tensorflow",
    ],
)

pytype_strict_library(
    name = "network_autoencoder",
    srcs = ["network_autoencoder.py"],
    srcs_version = "PY3",
    deps = [
        ":network_utils",
        "//third_party/py/tensorflow",
    ],
)

py_test(
    name = "network_autoencoder_test",
    srcs = ["network_autoencoder_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":network_autoencoder",
        "//third_party/py/absl/testing:parameterized",
        "//third_party/py/tensorflow",
    ],
)

pytype_strict_library(
    name = "network_multilevel",
    srcs = ["network_multilevel.py"],
    srcs_version = "PY3",
    deps = [
        ":network_autoencoder",
        ":network_imnet",
        ":network_utils",
        "//third_party/py/tensorflow",
    ],
)

py_test(
    name = "network_multilevel_test",
    srcs = ["network_multilevel_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":network_multilevel",
        "//third_party/py/absl/testing:parameterized",
        "//third_party/py/tensorflow",
    ],
)

pytype_strict_library(
    name = "point_sampler_lib",
    srcs = ["point_sampler_lib.py"],
    srcs_version = "PY3",
    deps = [
        "//third_party/py/tensorflow",
        "//vr/perception/volume_compression/mdif/utils:point_sampling",
    ],
)

py_test(
    name = "point_sampler_lib_test",
    srcs = ["point_sampler_lib_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":point_sampler_lib",
        "//third_party/py/absl/testing:parameterized",
        "//third_party/py/tensorflow",
    ],
)

pytype_strict_library(
    name = "loss_lib",
    srcs = ["loss_lib.py"],
    srcs_version = "PY3",
    deps = [
        "//third_party/py/tensorflow",
        "//vr/perception/volume_compression/mdif/utils:misc_utils",
    ],
)

py_test(
    name = "loss_lib_test",
    srcs = ["loss_lib_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":loss_lib",
        "//third_party/py/absl/testing:parameterized",
        "//third_party/py/tensorflow",
    ],
)

pytype_strict_library(
    name = "network_pipeline",
    srcs = ["network_pipeline.py"],
    srcs_version = "PY3",
    deps = [
        ":network_autoencoder",
        ":network_multilevel",
        ":network_utils",
        ":point_sampler_lib",
        "//third_party/py/gin/tf",
        "//third_party/py/tensorflow",
        "//third_party/py/tensorflow_addons",
        "//third_party/py/tensorflow_graphics/math/interpolation:trilinear",
        "//vr/perception/volume_compression/mdif/utils:chamfer_distance",
        "//vr/perception/volume_compression/mdif/utils:grid_utils",
        "//vr/perception/volume_compression/mdif/utils:mesh_utils",
        "//vr/perception/volume_compression/mdif/utils:metric_utils",
        "//vr/perception/volume_compression/mdif/utils:misc_utils",
        "//vr/perception/volume_compression/mdif/utils:point_sampling",
    ],
)

py_test(
    name = "network_pipeline_test",
    srcs = ["network_pipeline_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":network_pipeline",
        "//third_party/py/absl/testing:parameterized",
        "//third_party/py/tensorflow",
    ],
)

pytype_strict_library(
    name = "dataset_lib",
    srcs = ["dataset_lib.py"],
    srcs_version = "PY3",
    deps = [
        "//pyglib:gfile",
        "//third_party/py/gin/tf",
        "//third_party/py/numpy",
        "//third_party/py/tensorflow:tensorflow_google",
    ],
)

py_test(
    name = "dataset_lib_test",
    srcs = ["dataset_lib_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":dataset_lib",
        "//sstable/python:pysstable",
        "//sstable/python:pywrapsstable",
        "//third_party/py/numpy",
        "//third_party/py/tensorflow:tensorflow_google",
    ],
)
