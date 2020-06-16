from typing import Dict

class AppConfig():
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