from typing import List, Optional
import os
import subprocess
import tempfile
from ppcgrader.logging import log_command


class RunnerOutput:
    def __init__(
        self,
        run_successful: bool,
        timed_out: bool,
        stdout: str,
        stderr: str,
        timeout: Optional[float],
        time: Optional[float] = None,
        errors: Optional[bool] = None,
        input_data=None,
        output_data=None,
        output_errors=None,
        statistics=None,
    ):
        self.run_successful = run_successful
        self.timed_out = timed_out
        self.stdout = stdout
        self.stderr = stderr
        self.timeout = timeout
        self.time = time
        self.errors = errors
        self.input_data = input_data
        self.output_data = output_data
        self.output_errors = output_errors
        self.statistics = statistics

    def is_success(self):
        return self.run_successful

    def is_timed_out(self):
        return self.timed_out


class AsanRunnerOutput(RunnerOutput):
    def __init__(
        self,
        run_successful: bool,
        timed_out: bool,
        stdout: str,
        stderr: str,
        timeout: Optional[float],
        asanoutput: str,
        time: Optional[float] = None,
        errors: Optional[bool] = None,
        input_data=None,
        output_data=None,
        output_errors=None,
        statistics=None,
    ):
        self.run_successful = run_successful
        self.timed_out = timed_out
        self.stdout = stdout
        self.stderr = stderr
        self.timeout = timeout
        self.time = time
        self.errors = errors
        self.asanoutput = asanoutput
        self.input_data = input_data
        self.output_data = output_data
        self.output_errors = output_errors
        self.statistics = statistics


class MemcheckRunnerOutput(RunnerOutput):
    def __init__(
        self,
        run_successful: bool,
        timed_out: bool,
        stdout: str,
        stderr: str,
        timeout: Optional[float],
        memcheckoutput: Optional[str],
        time: Optional[float] = None,
        errors: Optional[bool] = None,
        input_data=None,
        output_data=None,
        output_errors=None,
        statistics=None,
    ):
        self.run_successful = run_successful
        self.timed_out = timed_out
        self.stdout = stdout
        self.stderr = stderr
        self.timeout = timeout
        self.time = time
        self.errors = errors
        self.memcheckoutput = memcheckoutput
        self.input_data = input_data
        self.output_data = output_data
        self.output_errors = output_errors
        self.statistics = statistics


class Runner:
    def run(self, config, args: List[str],
            timeout: Optional[float]) -> RunnerOutput:
        env = os.environ.copy()

        ppc_output_read, ppc_output_write = os.pipe()
        env['PPC_OUTPUT'] = str(ppc_output_write)

        env['PPC_PERF'] = 'default'

        log_command(args)
        process = subprocess.Popen(args,
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE,
                                   stdin=subprocess.DEVNULL,
                                   env=env,
                                   encoding='utf-8',
                                   errors='utf-8',
                                   pass_fds=(ppc_output_write, ))
        os.close(ppc_output_write)

        try:
            stdout, stderr = process.communicate(None, timeout=timeout)
            run_successful = process.returncode == 0
            timed_out = False
        except subprocess.TimeoutExpired:
            run_successful = False
            timed_out = True
            try:
                # Ask nicely to terminate before killing
                process.terminate()
                stdout, stderr = process.communicate(None, timeout=1)
            except subprocess.TimeoutExpired:
                process.kill()
                stdout, stderr = process.communicate(None)

        if run_successful:
            output = os.fdopen(ppc_output_read, 'r').read()
            results = config.parse_output(output)
        else:
            os.close(ppc_output_read)
            results = []

        return RunnerOutput(run_successful, timed_out, stdout, stderr, timeout,
                            *results)


class AsanRunner(Runner):
    def __init__(self):
        self.env = {}

    def run(self, config, args: List[str], timeout: float) -> AsanRunnerOutput:
        env = os.environ.copy()
        env.update(self.env)
        if 'ASAN_OPTIONS' in env:
            env['ASAN_OPTIONS'] += ':log_path=/tmp/asan.log'
        else:
            env['ASAN_OPTIONS'] = 'log_path=/tmp/asan.log'

        ppc_output_read, ppc_output_write = os.pipe()
        env['PPC_OUTPUT'] = str(ppc_output_write)

        log_command(args)
        process = subprocess.Popen(args,
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE,
                                   stdin=subprocess.DEVNULL,
                                   env=env,
                                   encoding='utf-8',
                                   errors='utf-8',
                                   pass_fds=(ppc_output_write, ))
        os.close(ppc_output_write)

        try:
            stdout, stderr = process.communicate(None, timeout=timeout)
            run_successful = process.returncode == 0
            timed_out = False
        except subprocess.TimeoutExpired:
            timed_out = True
            run_successful = False
            try:
                # Ask nicely to terminate before killing
                process.terminate()
                stdout, stderr = process.communicate(None, timeout=1)
            except subprocess.TimeoutExpired:
                process.kill()
                stdout, stderr = process.communicate(None)

        # Read Asan output
        if os.path.exists(f"/tmp/asan.log.{process.pid}"):
            with open(f"/tmp/asan.log.{process.pid}") as f:
                asanoutput = f.read()
            # Try to delete the output file
            os.remove(f"/tmp/asan.log.{process.pid}")
        else:
            asanoutput = None

        if run_successful:
            output = os.fdopen(ppc_output_read, 'r').read()
            results = config.parse_output(output)
        else:
            os.close(ppc_output_read)
            results = []

        return AsanRunnerOutput(run_successful, timed_out, stdout, stderr,
                                timeout, asanoutput, *results)


class MemcheckRunner(Runner):
    def __init__(self, tool: str):
        self.env = {}
        self.tool = tool

    def run(self, config, args: List[str],
            timeout: float) -> MemcheckRunnerOutput:
        env = os.environ.copy()
        env.update(self.env)

        ppc_output_read, ppc_output_write = os.pipe()
        env['PPC_OUTPUT'] = str(ppc_output_write)

        # Run with memcheck
        memcheck_output_file = tempfile.NamedTemporaryFile('r')
        args = [
            'cuda-memcheck',
            '--tool',
            self.tool,
            '--log-file',
            memcheck_output_file.name,
            '--error-exitcode',
            '1',
            '--prefix',
            ' ',
            '--print-limit',
            '1000',
            *(['--leak-check', 'full'] if self.tool == 'memcheck' else []),
            '--',
        ] + args

        log_command(args)
        process = subprocess.Popen(args,
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE,
                                   stdin=subprocess.DEVNULL,
                                   env=env,
                                   encoding='utf-8',
                                   errors='utf-8',
                                   pass_fds=(ppc_output_write, ))
        os.close(ppc_output_write)

        try:
            stdout, stderr = process.communicate(None, timeout=timeout)
            run_successful = process.returncode == 0
            timed_out = False
        except subprocess.TimeoutExpired:
            timed_out = True
            run_successful = False
            try:
                # Ask nicely to terminate before killing
                process.terminate()
                stdout, stderr = process.communicate(None, timeout=1)
            except subprocess.TimeoutExpired:
                process.kill()
                stdout, stderr = process.communicate(None)

        # Read memcheck output
        memcheckoutput = memcheck_output_file.read()
        no_outputs = [
            '  CUDA-MEMCHECK\n  LEAK SUMMARY: 0 bytes leaked in 0 allocations\n  ERROR SUMMARY: 0 errors\n',
            '  CUDA-MEMCHECK\n  RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings) \n',
            '  CUDA-MEMCHECK\n  ERROR SUMMARY: 0 errors\n',
        ]
        if memcheckoutput in no_outputs:
            memcheckoutput = None

        if run_successful:
            output = os.fdopen(ppc_output_read, 'r').read()
            results = config.parse_output(output)
        else:
            os.close(ppc_output_read)
            results = []

        return MemcheckRunnerOutput(run_successful, timed_out, stdout, stderr,
                                    timeout, memcheckoutput, *results)


class TsanRunner(AsanRunner):
    pass
