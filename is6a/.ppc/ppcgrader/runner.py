import csv
from typing import Dict, List, Optional, Any, Tuple
import io
import os
import re
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


class NvprofRunnerOutput(RunnerOutput):
    def __init__(self, nvprof_raw, nvprof, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.nvprof_raw = nvprof_raw
        self.nvprof = nvprof


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


class NvprofRunner(Runner):
    def run(self, config, args: List[str],
            timeout: Optional[float]) -> RunnerOutput:
        env = os.environ.copy()

        ppc_output_read, ppc_output_write = os.pipe()
        env['PPC_OUTPUT'] = str(ppc_output_write)

        env['PPC_PERF'] = 'default'

        # Run with nvprof
        nvprof_output_file = tempfile.NamedTemporaryFile('r')
        args = [
            'nvprof',
            '--csv',
            '--log-file',
            nvprof_output_file.name,
            '--normalized-time-unit',
            's',
            '--print-gpu-trace',
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
            run_successful = False
            timed_out = True
            try:
                # Ask nicely to terminate before killing
                process.terminate()
                stdout, stderr = process.communicate(None, timeout=1)
            except subprocess.TimeoutExpired:
                process.kill()
                stdout, stderr = process.communicate(None)

        # Read nvprof output
        def get_scale(unit: str, base=1000) -> float:
            if unit == 'm':
                return base**-1
            elif unit == 'K':
                return base**1
            elif unit == 'M':
                return base**2
            elif unit == 'G':
                return base**3
            elif unit == 'T':
                return base**4
            elif unit == 'P':
                return base**5
            return 1

        def split_unit(unit: str) -> Tuple[str, float]:
            if unit.endswith('B'):
                scale = get_scale(unit[:-1], 1024)
                return ('bytes', scale)
            elif unit.endswith('B/s'):
                scale = get_scale(unit[:-3], 1024)
                return ('bytes/s', scale)
            elif unit.endswith('s'):
                scale = get_scale(unit[:-1])
                return ('s', scale)
            return ('', 1.)

        def normalize_units(row: Dict[str, Any], units: Dict[str, str]):
            normalized = {}
            for k, v in row.items():
                suffix, scale = split_unit(units[k])
                if suffix or scale != 1:
                    k = k + ' ' + suffix
                    if v:
                        v = float(v) * scale
                    else:
                        v = None
                normalized[k] = v
            return normalized

        nvprofoutput_raw = nvprof_output_file.read()
        nvprof_statistics = {}
        try:
            # First line should look something like
            # ==[PID]== NVPROF is profiling process [PID], command: [some command]
            # Let's parse the PID part and use ==PID== as a comment marker in the future
            first_line = nvprofoutput_raw.split('\n', 1)[0]
            m = re.match(r'^==(\d+)== (.*)', first_line)
            pid = m.group(1)
            section = m.group(2)
            r = re.compile(fr'^=={re.escape(pid)}== (.*?)$', re.MULTILINE)
            pos = m.end()
            while True:
                m = r.search(nvprofoutput_raw, pos)
                if m is None:
                    end = -1
                else:
                    end = m.start()

                content = nvprofoutput_raw[pos:end].strip()

                if section == 'Profiling result:':
                    gpu_trace = list(
                        csv.DictReader(io.StringIO(content),
                                       dialect=csv.unix_dialect))
                    # Normalize units
                    units = gpu_trace[0]
                    gpu_trace = gpu_trace[1:]

                    if len(gpu_trace) <= 100:
                        gpu_trace = [
                            normalize_units(row, units) for row in gpu_trace
                        ]
                        nvprof_statistics['gpu_trace'] = gpu_trace
                    else:
                        nvprof_statistics['gpu_trace'] = None
                        nvprof_statistics[
                            'gpu_trace_message'] = "Too long GPU trace."

                if m is None:
                    break
                section = m.group(1)
                pos = m.end()
        except:
            # Parsing nvprof output failed
            pass

        if run_successful:
            output = os.fdopen(ppc_output_read, 'r').read()
            results = config.parse_output(output)
        else:
            os.close(ppc_output_read)
            results = []

        return NvprofRunnerOutput(nvprofoutput_raw, nvprof_statistics,
                                  run_successful, timed_out, stdout, stderr,
                                  timeout, *results)


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
