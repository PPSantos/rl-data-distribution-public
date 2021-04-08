"""
    Custom Default logger.
    See acme.utils.loggers for more details.
"""
import os
import csv
import time
from absl import logging

from acme.utils.loggers import aggregators
from acme.utils.loggers import base
# from acme.utils.loggers import csv
# from acme.utils.loggers import filters
from acme.utils.loggers import tf_summary

def make_default_logger(
    directory: str,
    label: str,
    save_csv: bool = True,
    # time_delta: float = 1.0,
    ) -> base.Logger:
    """
    Make a default Acme logger.
    Args:
    directory: Log directory.
    label: Name to give to the logger.
    save_csv: Whether to log csv file.
    time_delta: Time (in seconds) between logging events.
    Returns:
    A logger (pipe) object that responds to logger.write(some_dict).
    """
    tb_logger = tf_summary.TFSummaryLogger(logdir=directory, label=label)

    loggers = [tb_logger]
    if save_csv:
        loggers.append(CSVLogger(directory=directory, label=label))

    logger = aggregators.Dispatcher(loggers)
    # logger = filters.NoneFilter(logger)
    # logger = filters.TimeFilter(logger, time_delta)
    return logger


class CSVLogger(base.Logger):
  """Standard CSV logger."""

  _open = open

  def __init__(self,
               directory: str = '~/acme',
               label: str = '',
               time_delta: float = 0.):
    # directory = paths.process_path(directory, 'logs', label, add_uid=True)
    self._file_path = os.path.join(directory, f'{label}.csv')
    logging.info('Logging to %s', self._file_path)
    self._time = time.time()
    self._time_delta = time_delta
    self._header_exists = False

  def write(self, data: base.LoggingData):
    """Writes a `data` into a row of comma-separated values."""

    # Only log if `time_delta` seconds have passed since last logging event.
    now = time.time()
    if now - self._time < self._time_delta:
      return
    self._time = now

    # Append row to CSV.
    with self._open(self._file_path, mode='a') as f:
      keys = sorted(data.keys())
      writer = csv.DictWriter(f, fieldnames=keys)
      if not self._header_exists:
        # Only write the column headers once.
        writer.writeheader()
        self._header_exists = True
      writer.writerow(data)

  @property
  def file_path(self) -> str:
    return self._file_path