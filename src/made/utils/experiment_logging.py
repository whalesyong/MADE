from __future__ import annotations

import logging
import sys
import threading
from pathlib import Path


_LOG_FORMAT = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
_EXCEPTION_HOOKS_INSTALLED = False
_ORIGINAL_SYS_EXCEPTHOOK = sys.excepthook
_ORIGINAL_THREAD_EXCEPTHOOK = getattr(threading, "excepthook", None)


def _install_exception_hooks() -> None:
    """Mirror unhandled exceptions into the configured root logger."""
    global _EXCEPTION_HOOKS_INSTALLED
    if _EXCEPTION_HOOKS_INSTALLED:
        return

    def _log_unhandled_exception(exc_type, exc_value, exc_traceback) -> None:
        if issubclass(exc_type, KeyboardInterrupt):
            _ORIGINAL_SYS_EXCEPTHOOK(exc_type, exc_value, exc_traceback)
            return
        logging.getLogger(__name__).critical(
            "Unhandled exception",
            exc_info=(exc_type, exc_value, exc_traceback),
        )
        _ORIGINAL_SYS_EXCEPTHOOK(exc_type, exc_value, exc_traceback)

    def _log_thread_exception(args) -> None:
        if issubclass(args.exc_type, KeyboardInterrupt):
            if _ORIGINAL_THREAD_EXCEPTHOOK is not None:
                _ORIGINAL_THREAD_EXCEPTHOOK(args)
            return
        logging.getLogger(__name__).critical(
            "Unhandled thread exception",
            exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
        )
        if _ORIGINAL_THREAD_EXCEPTHOOK is not None:
            _ORIGINAL_THREAD_EXCEPTHOOK(args)

    sys.excepthook = _log_unhandled_exception
    if _ORIGINAL_THREAD_EXCEPTHOOK is not None:
        threading.excepthook = _log_thread_exception
    _EXCEPTION_HOOKS_INSTALLED = True


def configure_experiment_logging(
    output_dir: str | Path | None,
    log_level: str = "INFO",
    log_filename: str = "experiment.log",
    *,
    force: bool = False,
) -> Path | None:
    """Configure console + experiment file logging for the current process."""
    level = getattr(logging, str(log_level).upper(), logging.INFO)
    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)
    root_logger = logging.getLogger()

    log_path: Path | None = None
    if output_dir is not None:
        log_path = Path(output_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        log_path = log_path / log_filename

    if force:
        for handler in list(root_logger.handlers):
            root_logger.removeHandler(handler)
            try:
                handler.close()
            except Exception:
                pass

    root_logger.setLevel(level)

    has_stream_handler = any(
        getattr(handler, "_made_stream_handler", False)
        for handler in root_logger.handlers
    )
    if not has_stream_handler:
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(level)
        stream_handler.setFormatter(formatter)
        stream_handler._made_stream_handler = True  # type: ignore[attr-defined]
        root_logger.addHandler(stream_handler)

    if log_path is not None:
        resolved_log_path = str(log_path.resolve())
        has_file_handler = any(
            isinstance(handler, logging.FileHandler)
            and getattr(handler, "_made_log_path", None) == resolved_log_path
            for handler in root_logger.handlers
        )
        if not has_file_handler:
            file_handler = logging.FileHandler(log_path, encoding="utf-8")
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            file_handler._made_log_path = resolved_log_path  # type: ignore[attr-defined]
            root_logger.addHandler(file_handler)

    for handler in root_logger.handlers:
        handler.setLevel(level)

    _install_exception_hooks()
    return log_path
