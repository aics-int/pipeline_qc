from typing import Dict

class AppConfig:
    """
    Application configuration interface
    """
    def __init__(self, config: Dict): 
        self._config: Dict = config
    
    @property
    def fms_host(self) -> str:
        return self._config["fms_host"]

    @property
    def fms_port(self) -> int:
        return int(self._config["fms_port"])

    @property
    def fms_timeout_in_seconds(self) -> int:
        return int(self._config["fms_timeout_in_seconds"])
        
    @property
    def labkey_host(self) -> str:
        return self._config["labkey_host"]

    @property
    def labkey_port(self) -> int:
        return int(self._config["labkey_port"])


class GpuClusterConfig:
    """
    GPU Cluster configuration interface (for distributed runs)
    """
    def __init__(self, gpu:str, config: Dict): 
        self._gpu = gpu
        self._config: Dict = config

    @property
    def gpu(self) -> str:
        """
        Gpu type/name
        """
        return self._gpu
        
    @property
    def partition(self) -> str:
        """
        Cluster partition
        """
        return self._config.get("partition", "aics_gpu_general")
    
    @property
    def cluster_size(self) -> int:
        """
        Number of nodes to scale the cluster to (should be the number of available GPUs)
        """
        return int(self._config["cluster_size"])

    @property
    def worker_memory_limit(self) -> str:
        """
        Memory limit per worker/job
        """
        return self._config["worker_memory_limit"]

    @property
    def worker_time_limit(self) -> str:
        """
        Time limit for a cluster worker/node to stay alive
        """
        return self._config["worker_time_limit"]

    