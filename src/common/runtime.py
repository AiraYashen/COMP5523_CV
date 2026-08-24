from __future__ import annotations

import os
import warnings


def configure_runtime() -> None:
    os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    os.environ.setdefault("GLOG_minloglevel", "2")

    warnings.filterwarnings("ignore", message=".*LibreSSL.*")
    warnings.filterwarnings("ignore", message=".*return_token_timestamps.*")
    warnings.filterwarnings("ignore", message=".*generation flags are not valid.*")
    warnings.filterwarnings(
        "ignore", message=".*labels.*post_process_grounded_object_detection.*"
    )
    warnings.filterwarnings(
        "ignore", message=".*use_fast=True will be the default behavior.*"
    )
    warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")

    try:
        from transformers import logging as transformers_logging

        transformers_logging.set_verbosity_error()
    except Exception:
        pass

    try:
        from absl import logging as absl_logging

        absl_logging.set_verbosity(absl_logging.ERROR)
        absl_logging.set_stderrthreshold("error")
    except Exception:
        pass
