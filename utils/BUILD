# Codes for general utility functions.

load("//devtools/python/blaze:pytype.bzl", "pytype_strict_library")

package(default_visibility = [
    "//vr/perception:__subpackages__",
])

pytype_strict_library(
    name = "point_sampling",
    srcs = ["point_sampling.py"],
    srcs_version = "PY3",
    deps = [
        "//third_party/py/tensorflow",
    ],
)

py_test(
    name = "point_sampling_test",
    srcs = ["point_sampling_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":point_sampling",
        "//third_party/py/absl/testing:parameterized",
        "//third_party/py/tensorflow",
    ],
)

pytype_strict_library(
    name = "grid_utils",
    srcs = ["grid_utils.py"],
    srcs_version = "PY3",
    deps = [
        "//third_party/py/tensorflow",
    ],
)

py_test(
    name = "grid_utils_test",
    srcs = ["grid_utils_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":grid_utils",
        "//third_party/py/tensorflow",
    ],
)

pytype_strict_library(
    name = "chamfer_distance",
    srcs = ["chamfer_distance.py"],
    srcs_version = "PY3",
    deps = [
        "//third_party/py/tensorflow",
    ],
)

py_test(
    name = "chamfer_distance_test",
    srcs = ["chamfer_distance_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":chamfer_distance",
        "//third_party/py/tensorflow",
    ],
)

pytype_strict_library(
    name = "metric_utils",
    srcs = ["metric_utils.py"],
    srcs_version = "PY3",
    deps = [
        "//third_party/py/tensorflow",
    ],
)

py_test(
    name = "metric_utils_test",
    srcs = ["metric_utils_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":metric_utils",
        "//third_party/py/tensorflow",
    ],
)

pytype_strict_library(
    name = "mesh_utils",
    srcs = ["mesh_utils.py"],
    srcs_version = "PY3",
    deps = [
        "//third_party/py/numpy",
        "//third_party/py/skimage",
        "//third_party/py/tensorflow",
    ],
)

py_test(
    name = "mesh_utils_test",
    srcs = ["mesh_utils_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":mesh_utils",
        "//third_party/py/tensorflow",
    ],
)

pytype_strict_library(
    name = "misc_utils",
    srcs = ["misc_utils.py"],
    srcs_version = "PY3",
    deps = [
        "//pyglib:gfile",
        "//third_party/py/numpy",
        "//third_party/py/tensorflow",
    ],
)

py_test(
    name = "misc_utils_test",
    srcs = ["misc_utils_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":misc_utils",
        "//pyglib:file_util",
        "//third_party/py/absl/testing:parameterized",
        "//third_party/py/numpy",
        "//third_party/py/tensorflow",
    ],
)
