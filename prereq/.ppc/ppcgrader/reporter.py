from typing import List, Dict, Optional, Union
import json
import sys

from ppcgrader.config import Config
from ppcgrader.compiler import Compiler, CompilerOutput
from ppcgrader.runner import RunnerOutput, AsanRunnerOutput, MemcheckRunnerOutput


def statistics_terminal(output):
    stat = output.statistics
    if stat is None:
        return None
    wallclock = stat.get('perf_wall_clock_ns', None)
    cputime = stat.get('perf_cpu_time_ns', None)
    instrs = stat.get('perf_instructions', None)
    cycles = stat.get('perf_cycles', None)
    branches = stat.get('perf_branches', None)
    branch_misses = stat.get('perf_branch_misses', None)
    if not wallclock:
        return None
    if wallclock < 1e7:
        return None
    r = ""
    if cputime:
        r += f'  Your code used {wallclock/1e9:.3f} sec of wallclock time, and {cputime/1e9:.3f} sec of CPU time\n'
        r += f'  ≈ you used {cputime/wallclock:.1f} simultaneous hardware threads on average\n\n'
    if cycles:
        r += f'  The total number of clock cycles was {cycles/1e9:.1f} billion\n'
        r += f'  ≈ CPU was running at {cycles/cputime:.1f} GHz\n\n'
    if instrs:
        r += f'  The CPU executed {instrs/1e9:.2f} billion machine-language instructions\n'
        r += f'  ≈ {instrs/wallclock:.2f} instructions per nanosecond\n'
        if cycles:
            r += f'  ≈ {instrs/cycles:.2f} instructions per clock cycle\n'
        r += '\n'
    if branches:
        r += f'  {branches/instrs*100:.1f}% of the instructions were branches\n'
        if branch_misses:
            r += f'  and {branch_misses/branches*100:.1f}% of them were mispredicted\n'
        r += '\n'
    r = r.rstrip()
    if len(r):
        return r
    else:
        return None


def _safe_json_dump(data):
    return json.dumps(
        json.loads(json.dumps(data), parse_constant=lambda s: str(s)))


class Reporter:
    class TestGroup:
        def compilation(self,
                        compiler: Compiler) -> 'Reporter.CompilationProxy':
            raise NotImplementedError()

        def test(self, test: str, output: RunnerOutput):
            raise NotImplementedError()

    class BenchmarkGroup:
        def compilation(self,
                        compiler: Compiler) -> 'Reporter.CompilationProxy':
            raise NotImplementedError()

        def benchmark(self, test: str, output: RunnerOutput):
            raise NotImplementedError()

    class AnalysisGroup:
        def compilation(self,
                        compiler: Compiler) -> 'Reporter.CompilationProxy':
            raise NotImplementedError()

        def analyze(self, output, success=True):
            raise NotImplementedError()

    class CompilationProxy:
        def compile(self, *args, **kwargs) -> 'CompilerOutput':
            raise NotImplementedError()

    def __init__(self, config: Config):
        self.config = config

    def test_group(self, name: str, tests: List[str]) -> 'TestGroup':
        raise NotImplementedError()

    def benchmark_group(self, name: str, tests: List[str]) -> 'BenchmarkGroup':
        raise NotImplementedError()

    def analysis_group(self, name: str) -> 'AnalysisGroup':
        raise NotImplementedError()

    def log(self, msg: str, kind=None):
        raise NotImplementedError()

    def finalize(self):
        raise NotImplementedError()


class TerminalReporter(Reporter):
    class TestGroup(Reporter.TestGroup):
        def __init__(self, reporter: 'TerminalReporter', tests: List[str]):
            self.reporter = reporter
            self.header_printed = False
            self.test_name_width = max(
                4, max(len(self._simplify_name(test)) for test in tests))

        def compilation(self,
                        compiler: Compiler) -> 'Reporter.CompilationProxy':
            return TerminalReporter.CompilationProxy(self.reporter, compiler)

        def test(self, test: str, output: RunnerOutput):
            if not self.header_printed:
                self.header_printed = True
                self.reporter.log(
                    f'{"test":<{self.test_name_width}}  {"time":>9}  {"result":6}',
                    'heading')
            if output.is_success():
                msg = "errors" if output.errors else "pass"
                self.reporter.log(
                    f'{self._simplify_name(test):<{self.test_name_width}}  {output.time:>8.3f}s  {msg:6}',
                    'error' if output.errors else 'pass')
            else:
                self.reporter.log(
                    f'{self._simplify_name(test):<{self.test_name_width}}  [failed]',
                    'error')

            if output.stdout:
                self.reporter.log_sep()
                self.reporter.log('Standard output:')
                self.reporter.log(output.stdout, 'output')
                self.reporter.log_sep()

            if output.stderr:
                self.reporter.log_sep()
                self.reporter.log('Standard error:')
                self.reporter.log(output.stderr, 'output')
                self.reporter.log_sep()

            if not output.is_success():
                self.reporter.log_sep()
                if output.is_timed_out():
                    self.reporter.log('It seems that your program timed out.')
                    self.reporter.log(
                        f'The test should have ran in less than {output.timeout} seconds.'
                    )
                    self.reporter.log(
                        'You can override allowed running time with --timeout [timeout in seconds]'
                    )
                else:
                    self.reporter.log(
                        'It seems that your program crashed unexpectedly.')
                self.reporter.log_sep()

            if isinstance(output, AsanRunnerOutput) and output.asanoutput:
                self.reporter.log_sep()
                self.reporter.log(
                    'AddressSanitizer reported the following errors:')
                self.reporter.log(output.asanoutput, 'output')
                self.reporter.log_sep()

            if isinstance(output,
                          MemcheckRunnerOutput) and output.memcheckoutput:
                self.reporter.log_sep()
                self.reporter.log('Memcheck reported the following errors:')
                self.reporter.log(output.memcheckoutput, 'output')
                self.reporter.log_sep()

            if output.errors and not self.reporter.config.ignore_errors:
                human_readable = self.reporter.config.explain_terminal(
                    output, self.reporter.color)
                if human_readable is not None:
                    self.reporter.log_sep()
                    self.reporter.log(human_readable, 'preformatted')
                    self.reporter.log_sep()

        def _simplify_name(self, test: str):
            # return os.path.basename(test)
            return test

    class BenchmarkGroup(Reporter.BenchmarkGroup):
        def __init__(self, reporter: 'TerminalReporter', tests: List[str]):
            self.reporter = reporter
            self.header_printed = False
            self.test_name_width = max(
                4, max(len(self._simplify_name(test)) for test in tests))

        def compilation(self,
                        compiler: Compiler) -> 'Reporter.CompilationProxy':
            return TerminalReporter.CompilationProxy(self.reporter, compiler)

        def benchmark(self, test: str, output: RunnerOutput):
            if not self.header_printed:
                self.header_printed = True
                self.reporter.log(
                    f'{"test":<{self.test_name_width}}  {"time":>9}  {"result":6}',
                    'heading')
            if output.is_success():
                msg = "errors" if output.errors else "pass"
                self.reporter.log(
                    f'{self._simplify_name(test):<{self.test_name_width}}  {output.time:>8.3f}s  {msg:6}',
                    'error' if output.errors else 'pass')
            else:
                self.reporter.log(
                    f'{self._simplify_name(test):<{self.test_name_width}}  [failed]',
                    'error')

            if output.is_success() and not output.errors:
                human_readable = statistics_terminal(output)
                if human_readable is not None:
                    self.reporter.log_sep()
                    self.reporter.log(human_readable, 'preformatted')
                    self.reporter.log_sep()

        def _simplify_name(self, test: str):
            # return os.path.basename(test)
            return test

    class AnalysisGroup(Reporter.AnalysisGroup):
        def __init__(self, name: str, reporter: 'TerminalReporter'):
            self.name = name
            self.reporter = reporter

        def compilation(self,
                        compiler: Compiler) -> 'Reporter.CompilationProxy':
            return TerminalReporter.CompilationProxy(self.reporter, compiler)

        def analyze(self, output, success=True):
            self.reporter.log(f'Output for {self.name}:', 'heading')
            self.reporter.log(output, 'output')

    class CompilationProxy(Reporter.CompilationProxy):
        def __init__(self, reporter: 'TerminalReporter', compiler: Compiler):
            self.reporter = reporter
            self.compiler = compiler

        def compile(self, *args, **kwargs) -> 'CompilerOutput':
            self.reporter.log('Compiling...')
            result = self.compiler.compile(*args, **kwargs)
            if result.stdout:
                self.reporter.log_sep()
                self.reporter.log('Compiler stdout:')
                self.reporter.log(result.stdout, 'output')
                self.reporter.log_sep()
            if result.stderr:
                self.reporter.log_sep()
                self.reporter.log('Compiler stderr:')
                self.reporter.log(result.stderr, 'output')
                self.reporter.log_sep()
            if result.is_success():
                self.reporter.log('Compiled')
            else:
                self.reporter.log('Compilation failed!', 'error')
            return result

    def __init__(self, config: Config):
        super().__init__(config)
        self.color = sys.stdout.isatty()
        self.sep_printed = False

    def test_group(self, name: str, tests: List[str]) -> 'TestGroup':
        return TerminalReporter.TestGroup(self, tests)

    def benchmark_group(self, name: str, tests: List[str]) -> 'BenchmarkGroup':
        return TerminalReporter.BenchmarkGroup(self, tests)

    def analysis_group(self, name: str) -> 'AnalysisGroup':
        return TerminalReporter.AnalysisGroup(name, self)

    def log_sep(self):
        if not self.sep_printed:
            print()
            self.sep_printed = True

    def log(self, msg: str, kind=None):
        msg = msg.rstrip()
        before, after = '', ''
        if kind is not None and self.color:
            reset = '\033[0m'
            before, after = {
                'title': ('\033[34;1m', reset),
                'heading': ('\033[1m', reset),
                'error': ('\033[31;1m', reset),
                'pass': ('\033[34m', reset),
                'command': ('\033[34m', reset),
                'output': ('\033[34m', reset),
            }.get(kind, ('', ''))
        print(before + msg + after)
        self.sep_printed = False

    def finalize(self):
        pass


def output_to_json(test: str, output: RunnerOutput,
                   benchmark: bool) -> Dict[str, str]:
    result = {
        'name': test,
        'test': open(test, 'r').read(),
        'success': output.is_success(),
    }
    if output.is_success():
        result['time'] = output.time
        result['errors'] = output.errors
        if output.errors:
            result['input'] = output.input_data
            result['output'] = output.output_data
            result['output_errors'] = output.output_errors
        if benchmark:
            result['statistics'] = output.statistics
    else:
        result['timed_out'] = output.is_timed_out()
    if isinstance(output, AsanRunnerOutput):
        result['asanoutput'] = output.asanoutput
    if isinstance(output, MemcheckRunnerOutput):
        result['memcheckoutput'] = output.memcheckoutput
    return result


class JsonReporter(Reporter):
    class TestGroup(Reporter.TestGroup):
        def __init__(self, name: str):
            self.name = name
            self.compiler_output = None
            self.tests = []

        def compilation(self,
                        compiler: Compiler) -> 'Reporter.CompilationProxy':
            if self.compiler_output is not None:
                raise RuntimeError(
                    'Must not compiler code twice in test group')
            self.compiler_output = {}
            return JsonReporter.CompilationProxy(self.compiler_output,
                                                 compiler)

        def test(self, test: str, output: RunnerOutput):
            self.tests.append(output_to_json(test, output, False))

        def is_success(self):
            return self.compiler_output['status'] == 0 and all(
                test['success'] and test['errors'] == 0 for test in self.tests)

        def to_json(self):
            return {
                'name': self.name,
                'compiler_output': self.compiler_output,
                'tests': self.tests,
            }

    class BenchmarkGroup(Reporter.BenchmarkGroup):
        def __init__(self, name: str):
            self.name = name
            self.compiler_output = None
            self.benchmarks = []

        def compilation(self,
                        compiler: Compiler) -> 'Reporter.CompilationProxy':
            if self.compiler_output is not None:
                raise RuntimeError(
                    'Must not compiler code twice in benchmark group')
            self.compiler_output = {}
            return JsonReporter.CompilationProxy(self.compiler_output,
                                                 compiler)

        def benchmark(self, test: str, output: RunnerOutput):
            self.benchmarks.append(output_to_json(test, output, True))

        def is_success(self):
            return self.compiler_output['status'] == 0 and all(
                benchmark['success'] and benchmark['errors'] == 0
                for benchmark in self.benchmarks)

        def to_json(self):
            return {
                'name': self.name,
                'compiler_output': self.compiler_output,
                'benchmarks': self.benchmarks,
            }

    class AnalysisGroup(Reporter.AnalysisGroup):
        def __init__(self, name: str):
            self.name = name
            self.compiler_output = None
            self.output = None
            self.success = False

        def compilation(self,
                        compiler: Compiler) -> 'Reporter.CompilationProxy':
            if self.compiler_output is not None:
                raise RuntimeError(
                    'Must not compiler code twice in analysis group')
            self.compiler_output = {}
            return JsonReporter.CompilationProxy(self.compiler_output,
                                                 compiler)

        def analyze(self, output, success=True):
            if self.output is not None:
                raise RuntimeError('Analysis must be recorded only once')
            self.output = output

        def is_success(self):
            return self.success

        def to_json(self):
            return {
                'name': self.name,
                'compiler_output': self.compiler_output,
                'output': self.output,
                'success': self.success,
            }

    class CompilationProxy(Reporter.CompilationProxy):
        def __init__(self, output: Dict[str, Union[str, int]],
                     compiler: Compiler):
            self.output = output
            self.compiler = compiler

        def compile(self, *args, **kwargs) -> 'CompilerOutput':
            result = self.compiler.compile(*args, **kwargs)
            self.output['status'] = result.returncode
            self.output['stdout'] = result.stdout
            self.output['stderr'] = result.stderr
            return result

    def __init__(self, config: Config):
        super().__init__(config)
        self.test_groups = []
        self.benchmark_groups = []
        self.analysis_groups = []

    def test_group(self, name: str, tests: List[str]) -> 'TestGroup':
        group = JsonReporter.TestGroup(name)
        self.test_groups.append(group)
        return group

    def benchmark_group(self, name: str, tests: List[str]) -> 'BenchmarkGroup':
        group = JsonReporter.BenchmarkGroup(name)
        self.benchmark_groups.append(group)
        return group

    def analysis_group(self, name: str) -> 'AnalysisGroup':
        group = JsonReporter.AnalysisGroup(name)
        self.analysis_groups.append(group)
        return group

    def log(self, msg: str, kind=None):
        pass  # No logging with json output

    def finalize(self):
        print(_safe_json_dump(self.to_json()))

    def to_json(self):
        return {
            'success':
            all(group.is_success() for group in self.test_groups)
            and all(group.is_success() for group in self.benchmark_groups)
            and all(group.is_success() for group in self.analysis_groups),
            'tests': [group.to_json() for group in self.test_groups],
            'benchmarks': [group.to_json() for group in self.benchmark_groups],
            'analyses': [group.to_json() for group in self.analysis_groups],
        }
