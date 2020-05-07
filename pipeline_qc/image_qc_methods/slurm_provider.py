import uuid
import re
import os

from dataclasses import dataclass
from typing import List, Optional
from pytz import utc
from datetime import datetime, timedelta
from tempfile import NamedTemporaryFile
from .ssh_client import SshClient


@dataclass
class SlurmJobHandle:
    """
    A handle for a running Slurm job
    Provides all the needed information to interface with a job during and after its execution
    """
    job_id: str
    submission_time_utc: datetime
    working_directory: str

    @property
    def output_file_path(self) -> str:
        """
        Path to the Slurm job output (log) file
        """
        return f"slurm-{self.job_id}.out"


class SlurmProvider():
    """
    Provider interface for Slurm cluster remote operations
    """
    # Constants
    SLURM_WORKDIR_ROOT = "/tmp"
    SLURM_USER_ENV_KEY = "AIRFLOW__AICS__SLURM__USERNAME"

    # Constructor
    def __init__(self, slurm_host: str, ssh_client: SshClient = SshClient()):
        if not slurm_host:
            raise AttributeError("slurm_host")

        slurm_user = os.environ.get(self.SLURM_USER_ENV_KEY)

        if slurm_user:
            self._slurm_host = f"{slurm_user}@{slurm_host}"
        else:
            self._slurm_host = slurm_host

        self._ssh_client = ssh_client

    # Public Methods
    def cancel_job(self, handle: SlurmJobHandle):
        """
        Cancel a Slurm job
        :param handle: the Slurm job handle
        """
        if handle is None:
            raise AttributeError("handle")

        self._ssh_client.ssh(self._slurm_host, f"scancel -f -Q {handle.job_id}")

    def get_job_state(self, handle: SlurmJobHandle) -> Optional[str]:
        """
        Get the state of the slurm batch job (running, pending, complete, failed, etc.)
        :param handle: the Slurm job handle
        :return: Return the state as an upper case string if found, None otherwise
        """

        if handle is None:
            raise AttributeError("handle")

        # use 'sacct' to determine running state of our submitted batch.
        # sacct -n -p --format=JobId,State,ExitCode,Start,End -j {jobid}
        # -P: parse-able (pipe delimit fields)
        # -n: no header
        # "Start" and "End" return times in UTC.
        # Unfortunately we need to track time. Due to config changes, the JobIDs were reset to start again at 1,
        # but the existing jobs were not cleared out. We need to ensure if we see COMPLETED - it is for the current
        # run of the job - its start date should later than the submission time.
        #
        # sacct -P -n --format=JobId,State,ExitCode,Start,End -j 313338
        #   313338_0|CANCELLED|0:0|2019-09-04T23:35:35|2019-09-05T00:18:22
        #   313338_0.0|CANCELLED|0:0|2019-09-04T23:35:36|2019-09-05T00:18:22
        #
        # sacct -P -n --format=JobId,State,ExitCode,Start,End -j 42126
        #   42126|RUNNING|0:0|2019-10-30T23:15:07|Unknown
        #
        # sacct -P -n --format=JobId,State,ExitCode,Start,End -j  35000
        #   35000|COMPLETED|0:0|2019-10-25T00:00:15|2019-10-25T00:13:07
        #   35000.batch|COMPLETED|0:0|2019-10-25T00:00:15|2019-10-25T00:13:07
        #   35000.0|COMPLETED|0:0|2019-10-25T00:00:16|2019-10-25T00:13:07
        process_state = None

        output = self._ssh_client.ssh(self._slurm_host,
                                      f"sacct -P -n --format=JobId,State,ExitCode,Start,End -j {handle.job_id}")

        if (len(output) > 0):
            for line in output:
                data = line.split('|')
                if len(data) == 5:
                    # We only care about the overall job status - hence looking for line with only the job ID
                    if data[0] == handle.job_id:
                        if data[3] != 'Unknown':  # PENDING states don't have a start time
                            start_date_utc = utc.localize(datetime.strptime(data[3], '%Y-%m-%dT%H:%M:%S'))

                        # The results are valid only if the start_date is greater than the submission date.
                        # Submission date is adjusted to account for small clock discrepancies between hpc nodes and airflow worker nodes
                        adjusted_submission_time = utc.localize(handle.submission_time_utc - timedelta(hours=1))
                        if data[3] == 'Unknown' or start_date_utc >= adjusted_submission_time:
                            process_state = data[1].upper()
                else:
                    raise RuntimeError(f'Received unexpected content from SLURM "sacct" command.  Expected 5 columns,'
                                       f'but received {len(data)}.  Output is {line}')
        else:
            self.log.warning(f'No job state found for job {handle.job_id}')

        return process_state

    def clean_environment(self, handle: SlurmJobHandle):
        """
        Cleans the working environment on Slurm
        :param handle: the Slurm job handle
        """
        if handle is None:
            raise AttributeError("handle")

        # remove workdir
        if self._validate_workdir(handle.working_directory):
            self.log.info(f"Removing job working directory at {handle.working_directory}")
            self._ssh_client.ssh(self._slurm_host, f"rm -rf {handle.working_directory}")
        else:
            self.log.warning(
                f"Can't remove job working directory at {handle.working_directory} (unexpected directory format)")

        # remove output file
        self.log.info(f"Removing job output file at {handle.output_file_path}")
        self._ssh_client.ssh(self._slurm_host, f"rm -f {handle.output_file_path}")

    def submit_slurm_job(self, workdir: str, sbatch_script: str) -> SlurmJobHandle:
        """
        Submit job to Slurm for processing
        :param: sbatch_script: sbatch script to run on Slurm
        :param: workdir: job working directory
        :return: handle for the submitted Slurm job
        """

        self.log.info('Submitting Slurm sbatch script for processing')

        # Upload script
        script_path = self._transfer_slurm_script(sbatch_script, workdir)

        # Submit job
        job_id = None
        submission_time_utc = datetime.utcnow()
        output = self._ssh_client.ssh(self._slurm_host, f"sbatch --output=slurm-%j.out {script_path}")

        if (len(output) > 0):
            for line in output:
                if line.startswith('Submitted batch job'):  # The output should have a line with the SLURM job ID
                    match = re.match('.*?([0-9]+)$', line)
                    if match:
                        job_id = match.group(1)

            if job_id:
                handle = SlurmJobHandle(job_id=job_id, submission_time_utc=submission_time_utc,
                                        working_directory=workdir)
                self.log.info(f'Slurm job [{job_id}] successfully submitted with job id [{handle.job_id}]')
                self.log.info(f'Job submission time (UTC): {handle.submission_time_utc}')
                self.log.info(f'Job working directory: {handle.working_directory}')
                self.log.info(f'Logging job outputs at: {handle.output_file_path}')
                return handle
            else:
                raise RuntimeError('Error submitting job - no SLURM job ID was found in the output')
        else:
            raise RuntimeError('Failed to get a result from sbatch submission.')

        return None

    def get_job_output(self, handle: SlurmJobHandle, start_line: int = 1, end_line="$") -> List[str]:
        """
        Get slurm job output lines for the given range
        :param handle: the Slurm job handle
        :param: start_line: line number (inclusive) to start from
        :param: end_line: line number (inclusive) to end on
        :return: job output / log lines
        """
        if handle is None:
            raise AttributeError("handle")

        sed_command = f"if [[ -e {handle.output_file_path} ]] ; then " \
                      f"sed -n -e '{start_line},{end_line}P' {handle.output_file_path};" \
                      f"fi"
        return self._ssh_client.ssh(self._slurm_host, sed_command)

    def create_workdir(self) -> str:
        """
        Create a new working directory on Slurm
        The directory is named using a random unique identifier to avoid collisions
        :return: the working directory absolute path
        """
        session_id = uuid.uuid4().hex
        workdir = f"{self.SLURM_WORKDIR_ROOT}/{session_id}"

        self.log.info(f"Creating job working directory at {workdir}")
        self._ssh_client.ssh(self._slurm_host, f"mkdir -p {workdir}")

        return workdir

    # Private Methods
    def _transfer_slurm_script(self, sbatch_script: str, working_directory: str) -> str:
        """
        Copy a sbatch script onto the Slurm host
        :param: sbatch_script: sbatch script to run on Slurm
        :param: working_directory: working directory to upload to on Slurm
        :return: remote path to the sbatch script
        """
        with NamedTemporaryFile("w+t") as tmp_file:
            self.log.info(f"created tmp script file at {tmp_file.name}")
            self.log.info(f"==script contents==")
            self.log.info(sbatch_script)
            self.log.info(f"==end script contents==")
            tmp_file.write(sbatch_script)
            tmp_file.seek(0)
            script_path = f"{working_directory}/sbatch.sh"
            self._ssh_client.scp(tmp_file.name, f"{self._slurm_host}:{script_path}")

        return script_path

    def _validate_workdir(self, workdir: str):
        """
        Validates a remote directory path.
        Used as a sanity check to ensure we don't break Slurm master by doing operations on / or other OS directories
        """
        if not workdir:
            return False
        if not workdir.startswith(self.SLURM_WORKDIR_ROOT):
            return False
        if len(workdir) != len(self.SLURM_WORKDIR_ROOT) + 33:  # expected format is {root_dir}/{32 character hex string}
            return False

        return True
