from __future__ import annotations

import logging
import multiprocessing as mp
from configparser import ConfigParser
from pathlib import Path

from dask.distributed import Client, LocalCluster
from dask_jobqueue import SLURMCluster

logger = logging.getLogger(__name__)


class DeviceConfig:

    """Specifies the execution device parameters"""

    __slots__ = "config"

    def __init__(self: DeviceConfig, config_filepath: Path) -> None:
        """Return the Device Config object."""
        self.config = ConfigParser().read(config_filepath)

    def set_config(self: DeviceConfig, config: ConfigParser) -> None:
        """Not implemented for this base class."""
        logger.error(
            "Set config should not be called directly from the DeviceConfigclass",
        )
        raise NotImplementedError(
            "Set config should not be called directly from the DeviceConfig class",
        )


class DaskConfig(DeviceConfig):

    """A config object which specifies information necessary to run Dask jobqueues."""

    __slots__ = ("mode", "jobs", "client", "backend")

    def __init__(self: DaskConfig, config_filepath: ConfigParser) -> None:
        """
        Store command line arguments into the config object.

        :param config: A loaded configparser object.
        """
        self.mode = None
        self.jobs = None
        self.client = None
        self.backend = None
        super(DaskConfig, self).__init__(config_filepath)

    def set_config(self: DaskConfig, config: ConfigParser) -> None:
        """
        Start dask cluster with configuration specified in the config object.

        Args:
        ----
        config: .ini file containing the dask device configuration information

        """
        if config["dask"]["cluster"].lower() == "slurm":
            self.mode: str = "cluster"
            cluster: SLURMCluster = SLURMCluster()
            self.jobs: int = int(config["dask"]["min_jobs"])
            cluster.scale(int(config["dask"]["max_jobs"]))
        else:
            self.mode: str = "local"
            self.jobs: int = mp.cpu_count() - 1
            cluster: LocalCluster = LocalCluster(self.jobs)
            logger.info("Starting with %s workers", str(len(self.client.ncores())))
        self.client: Client = Client(cluster)
        if config["dask"]["cluster"].lower() == "slurm":
            logger.info("Waiting for %s workers", str(self.jobs))
            self.client.wait_for_workers(self.jobs)
            self.jobs -= 2
            logger.info("Starting with %s workers", str(len(self.client.ncores())))
