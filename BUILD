# Tensorflow training/eval infrastructure.

load("//devtools/python/blaze:pytype.bzl", "pytype_strict_binary", "pytype_strict_library")

pytype_strict_library(
    name = "train_eval_lib",
    srcs = ["train_eval_lib.py"],
    srcs_version = "PY3",
    deps = [
        "//pyglib:gfile",
        "//third_party/py/absl/logging",
        "//third_party/py/gin/tf",
        "//third_party/py/tensorflow",
        "//vr/perception/volume_compression/mdif/model:dataset_lib",
        "//vr/perception/volume_compression/mdif/model:loss_lib",
        "//vr/perception/volume_compression/mdif/model:network_pipeline",
        "//vr/perception/volume_compression/mdif/utils:misc_utils",
    ],
)

py_test(
    name = "train_eval_lib_test",
    srcs = ["train_eval_lib_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":train_eval_lib",
        "//third_party/py/absl/testing:parameterized",
        "//third_party/py/tensorflow",
        "//vr/perception/volume_compression/mdif/model:loss_lib",
        "//vr/perception/volume_compression/mdif/model:network_pipeline",
    ],
)

pytype_strict_binary(
    name = "train_main_local",
    srcs = ["train_main_local.py"],
    data = ["//vr/perception/volume_compression/mdif/gin_config:gin_configs"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":train_eval_lib",
        "//third_party/py/absl:app",
        "//third_party/py/absl/flags",
        "//third_party/py/gin/tf",
    ],
)

pytype_strict_binary(
    name = "train_main_xm",
    srcs = ["train_main_xm.py"],
    data = ["//vr/perception/volume_compression/mdif/gin_config:gin_configs"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":train_eval_lib",
        "//learning/deepmind/xmanager2/client/google",
        "//perftools/gputools/profiler:gpuprof_lib",
        "//perftools/gputools/profiler:xprofilez_with_server",
        "//pyglib:file_util",
        "//third_party/py/absl:app",
        "//third_party/py/absl/flags",
        "//third_party/py/gin/tf",
    ],
)

pytype_strict_binary(
    name = "eval_main_local",
    srcs = ["eval_main_local.py"],
    data = ["//vr/perception/volume_compression/mdif/gin_config:gin_configs"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":train_eval_lib",
        "//third_party/py/absl:app",
        "//third_party/py/absl/flags",
        "//third_party/py/gin/tf",
    ],
)

pytype_strict_binary(
    name = "eval_main_xm",
    srcs = ["eval_main_xm.py"],
    data = ["//vr/perception/volume_compression/mdif/gin_config:gin_configs"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":train_eval_lib",
        "//learning/deepmind/xmanager2/client/google",
        "//perftools/gputools/profiler:gpuprof_lib",
        "//perftools/gputools/profiler:xprofilez_with_server",
        "//pyglib:file_util",
        "//third_party/py/absl:app",
        "//third_party/py/absl/flags",
        "//third_party/py/gin/tf",
    ],
)

pytype_strict_binary(
    name = "inference_main_local",
    srcs = ["inference_main_local.py"],
    data = ["//vr/perception/volume_compression/mdif/gin_config:gin_configs"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":train_eval_lib",
        "//third_party/py/absl:app",
        "//third_party/py/absl/flags",
        "//third_party/py/absl/logging",
        "//third_party/py/gin/tf",
        "//third_party/py/tensorflow",
    ],
)

pytype_strict_binary(
    name = "inference_main_xm",
    srcs = ["inference_main_xm.py"],
    data = ["//vr/perception/volume_compression/mdif/gin_config:gin_configs"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":train_eval_lib",
        "//learning/deepmind/xmanager2/client/google",
        "//perftools/gputools/profiler:gpuprof_lib",
        "//perftools/gputools/profiler:xprofilez_with_server",
        "//pyglib:gfile",
        "//third_party/py/absl:app",
        "//third_party/py/absl/flags",
        "//third_party/py/absl/logging",
        "//third_party/py/gin/tf",
        "//third_party/py/tensorflow",
    ],
)
