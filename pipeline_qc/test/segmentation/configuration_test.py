import pytest

from pipeline_qc.segmentation.configuration import AppConfig, GpuClusterConfig

class TestAppConfig:
    CONFIG = {
        "fms_host": "stg-aics.corp.alleninstitute.org",
        "fms_port": 80,
        "fms_timeout_in_seconds": 300,
        "labkey_host": "stg-aics.corp.alleninstitute.org",
        "labkey_port": 80
    }

    def test_app_config(self):
        #Arrange
        config = AppConfig(self.CONFIG)

        #Assert
        assert config.fms_host == "stg-aics.corp.alleninstitute.org"
        assert config.fms_port == 80
        assert config.fms_timeout_in_seconds == 300
        assert config.labkey_host == "stg-aics.corp.alleninstitute.org"
        assert config.labkey_port == 80

class TestGpuClusterConfig:
    CONFIG = {
        "partition": "aics_gpu_general",
        "worker_time_limit": "10:00:00",
        "gpus":{
            "gtx1080" :{
                "cluster_size": 4,
                "memory_limit": "50G",    
            }, 
            "titanx":{
                "cluster_size": 8,
                "memory_limit": "100G"
            },
            "titanxp":{
                "cluster_size": 6,
                "memory_limit": "55G"
            },
            "v100":{
                "cluster_size": 12,
                "memory_limit": "150G"
            }
        }
    }

    @pytest.mark.parametrize("gpu,expected_cluster_size,expected_memory_limit", 
                             [("gtx1080", 4, "50G"), 
                              ("titanx", 8, "100G"),
                              ("titanxp", 6, "55G"),
                              ("v100", 12, "150G")])
    def test_gpu_cluster_config(self, gpu, expected_cluster_size, expected_memory_limit):
        #Arrange
        config = GpuClusterConfig(gpu, self.CONFIG)

        #Assert
        assert config.gpu == gpu
        assert config.cluster_size == expected_cluster_size
        assert config.partition == "aics_gpu_general"
        assert config.worker_memory_limit == expected_memory_limit
        assert config.worker_time_limit == "10:00:00"

    def test_gpu_cluster_config_bad_gpu_throws_configuration_exception(self):
        #Assert
        with pytest.raises(Exception):
            config = GpuClusterConfig("badgpu", self.CONFIG)
