import random
import subprocess
import time
import re
from subprocess import PIPE, STDOUT, CalledProcessError, CompletedProcess
from typing import Callable, List


class SshClient():
    """
    Wrapper interface for SSH client operations
    """
    # This is added to the ssh/scp arguments to avoid the ssh key prompt since there are
    # non-interactive sessions. This also helps avoid issues when hosts get redeployed
    # which usually updates the host keys. Note that because UserKnowHostsFile gets
    # redirected to /dev/null, SSH/SCP will always have a warning on connecting.
    # Hence we have to cleanup the output - see _clean_result_output().
    SSH_NO_HOST_KEY_CHECK = ['-o', 'UserKnownHostsFile=/dev/null',
                             '-o', 'StrictHostKeyChecking=no',
                             '-T']
    SSH_SOCKET_FILE = '/tmp/aics_airflow_plugins.core.ssh_client.ssh.socket'
    SSH_ENABLE_MULTIPLEXING = ['-o', 'ControlMaster=auto',
                               '-o', 'ControlPersist=10m',
                               '-S', SSH_SOCKET_FILE]
    SSH_MAX_RETRIES = 20
    SSH_RETRY_WAIT_SECONDS = 2

    def ssh(self, host: str, command: str, check_exit: bool = True) -> List[str]:
        """
        Execute an SSH command
        :param: host: ssh host
        :param: command: full command to run
        :param: check_exit: if set to true, raise exception if exit code is not 0
        :return: output lines
        """
        if not command:
            raise AttributeError("command")

        return self._run_with_retry(lambda: subprocess.run(
                                    ['ssh'] + self.SSH_NO_HOST_KEY_CHECK + self.SSH_ENABLE_MULTIPLEXING + [host] + [command],
                                    stdout=PIPE, stderr=STDOUT, check=check_exit))

    def scp(self, source: str, target: str, check_exit: bool = True) -> List[str]:
        """
        Execute a secure copy (SCP)
        :param: source: source path
        :param: target: remote target path
        :param: check_exit: if set to true, raise exception if exit code is not 0
        :return: output lines
        """
        return self._run_with_retry(lambda: subprocess.run(
                                    ['scp'] + self.SSH_NO_HOST_KEY_CHECK + [source, target],
                                    stdout=PIPE, stderr=STDOUT, check=check_exit))

    def _run_with_retry(self, func: Callable) -> List[str]:
        """
        Run the given Callable wrapped with retry and error logic
        :param func: function / callable to run
        :return: process output lines
        """
        retry_count = 0
        while True:
            try:
                result: CompletedProcess = func()
                return self._clean_result_output(result.stdout)
            except CalledProcessError as cpe:
                output_error = self._clean_result_output(cpe.stdout)
                if cpe.returncode == 255 and \
                        any('ssh_exchange_identification' in line.strip() for line in output_error):
                    # Handle the ssh_exchange_identification error resulting from too many connections
                    # being attempted at once - in the context of many Airflow dag runs using this
                    if retry_count >= self.SSH_MAX_RETRIES:
                        for line in output_error:
                            self.log.error(f"StdOut/Err: {line}")
                        raise RuntimeError("Too many retries on ssh_exchange_identification errors")
                else:
                    self.log.error("Error during ssh call")
                    self.log.error(str(cpe))
                    for line in output_error:
                        self.log.error(f"StdOut/Err: {line}")
                    raise RuntimeError("Ssh call failed") from cpe
            retry_count += 1
            self._wait(self.SSH_RETRY_WAIT_SECONDS, jitter_limit=0.5)

    def _wait(self, duration: int, jitter_limit: int = 2) -> None:
        """
        Wait for the given duration and add a bit of jitter to the wait to avoid concurrent calls
        :param: duration: Number of seconds to wait
        :param: jitter_limit: The variation limit.
        """
        wait_time = random.uniform(min(duration - jitter_limit, 0.5), duration + jitter_limit)
        time.sleep(wait_time)

    def _clean_result_output(self, output_bytes) -> List[str]:
        """
        Results from SSH come out as a bytestring. This converts the output to a string,
        then splits it into a list of lines. It also removes the line with the warning
        about adding the host to the list known hosts.
        :param: output_bytes: stdout output byte array
        :return: A list of strings corresponding to lines in stdout
        """
        if output_bytes is None:
            return []

        # Remove the expected and unwanted warning
        cleaned = re.sub(r"^Warning\: Permanently added.*?\r\n", "", str(output_bytes, 'utf-8'))
        lines = cleaned.split("\n")
        # This heuristic is necessary since the sed in drain logs will always return an new line
        if len(lines[-1]) == 0:
            lines = lines[:-1]
        return lines