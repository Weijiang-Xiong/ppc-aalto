from typing import List, Optional
import glob
import os
import sys

from ppcgrader.config import Config
from ppcgrader.runner import Runner, AsanRunner, TsanRunner, MemcheckRunner, NvprofRunner
from ppcgrader.compiler import Compiler, ClangCompiler, NvccCompiler, find_clang_compiler
from ppcgrader.reporter import Reporter

MAX_ASSEMBLY_OUTPUT = 600000  # Bytes


def parse_timeout(file: str, timeout: Optional[float]) -> Optional[float]:
    if timeout:
        return timeout

    first_line = open(file, 'r').readline().split(' ')
    if first_line[0] == "timeout":
        return float(first_line[1])

    return None


def run_test(compiler: Compiler,
             runner: Runner,
             reporter: Reporter,
             name: str,
             tests: List[str],
             config: Config,
             timeout: Optional[float] = None) -> bool:
    rep = reporter.test_group(name, tests)
    output = rep.compilation(
        compiler.add_source(config.tester).add_source(
            config.source)).compile(out_file=config.binary)
    if not output.is_success():
        return False

    for test in tests:
        runner_output = runner.run(config,
                                   config.test_command(test),
                                   timeout=parse_timeout(test, timeout))
        rep.test(test, runner_output)
        if not config.ignore_errors:
            if runner_output.errors or not runner_output.run_successful:
                return False

    return True


def run_benchmark(compiler: Compiler,
                  runner: Runner,
                  reporter: Reporter,
                  name: str,
                  tests: List[str],
                  config: Config,
                  timeout: Optional[float] = None) -> bool:
    rep = reporter.benchmark_group(name, tests)
    output = rep.compilation(
        compiler.add_source(config.tester).add_source(
            config.source)).compile(out_file=config.binary)
    if not output.is_success():
        return False

    for test in tests:
        runner_output = runner.run(config,
                                   config.benchmark_command(test),
                                   timeout=parse_timeout(test, timeout))
        rep.benchmark(test, runner_output)
        if runner_output.errors or not runner_output.run_successful:
            return False

    return True


def run_test_plain(compiler: Optional[Compiler],
                   reporter: Reporter,
                   tests: List[str],
                   config: Config,
                   timeout: Optional[float] = None) -> bool:
    reporter.log('Running tests', 'title')
    runner = Runner()
    compiler = config.find_compiler() if compiler is None else compiler
    if compiler is None:
        sys.exit("I'm sorry, I could not find a suitable compiler.")
    compiler = config.common_flags(compiler).add_flag('-O3').add_flag('-g')
    orig_tests = tests
    tests = expand_glob(tests, ['tests/*', 'benchmarks/*'])
    if not tests:
        no_tests_error(orig_tests, ['tests', 'benchmarks'])

    return run_test(compiler=compiler,
                    runner=runner,
                    reporter=reporter,
                    name='test-plain',
                    tests=tests,
                    config=config,
                    timeout=timeout)


def run_test_asan(compiler: Optional[Compiler],
                  reporter: Reporter,
                  tests: List[str],
                  config: Config,
                  timeout: Optional[float] = None) -> bool:
    reporter.log('Running tests with address sanitizer', 'title')
    runner = AsanRunner()
    compiler = config.find_compiler() if compiler is None else compiler
    if compiler is None:
        sys.exit("I'm sorry, I could not find a suitable compiler.")
    if isinstance(compiler, NvccCompiler):
        compiler = config.common_flags(compiler).add_flag(
            '-Xcompiler',
            '"-fsanitize=address"').add_flag('-Xcompiler',
                                             '"-fsanitize=undefined"')
        runner.env[
            'ASAN_OPTIONS'] = 'protect_shadow_gap=0:replace_intrin=0:detect_leaks=0'
    else:
        compiler = config.common_flags(compiler).add_flag(
            '-fsanitize=address').add_flag('-fsanitize=undefined')
    compiler = compiler.add_flag('-g')
    orig_tests = tests
    tests = expand_glob(tests, ['tests/*'])
    if not tests:
        no_tests_error(orig_tests, ['tests'])

    return run_test(compiler=compiler,
                    runner=runner,
                    reporter=reporter,
                    name='test-asan',
                    tests=tests,
                    config=config,
                    timeout=timeout)


def run_test_memcheck(tool: str,
                      compiler: Optional[Compiler],
                      reporter: Reporter,
                      tests: List[str],
                      config: Config,
                      timeout: Optional[float] = None) -> bool:
    reporter.log(f'Running tests with cuda-memcheck --tool {tool}', 'title')
    runner = MemcheckRunner(tool)
    compiler = config.find_compiler() if compiler is None else compiler
    if compiler is None:
        raise RuntimeError("Could not find a suitable compiler")
    compiler = config.common_flags(compiler).add_flag('-O3').add_flag(
        '-g').add_flag('-Xcompiler', '-rdynamic').add_flag('-lineinfo')
    orig_tests = tests
    tests = expand_glob(tests, ['tests/*memcheck*']) or expand_glob(
        tests, ['tests/*medium*'])
    if not tests:
        no_tests_error(orig_tests, ['tests'])

    return run_test(compiler=compiler,
                    runner=runner,
                    reporter=reporter,
                    name=f'test-memcheck-{tool}',
                    tests=tests,
                    config=config,
                    timeout=timeout)


def run_test_memcheck_memcheck(*args, **kwargs):
    return run_test_memcheck(tool='memcheck', *args, **kwargs)


def run_test_memcheck_racecheck(*args, **kwargs):
    return run_test_memcheck(tool='racecheck', *args, **kwargs)


def run_test_memcheck_initcheck(*args, **kwargs):
    return run_test_memcheck(tool='initcheck', *args, **kwargs)


def run_test_memcheck_synccheck(*args, **kwargs):
    return run_test_memcheck(tool='synccheck', *args, **kwargs)


def run_test_uninit(compiler: Optional[Compiler],
                    reporter: Reporter,
                    tests: List[str],
                    config: Config,
                    timeout: Optional[float] = None) -> bool:
    reporter.log('Running tests with uninitialized variable check', 'title')
    runner = Runner()
    compiler = find_clang_compiler() if compiler is None else compiler
    if compiler is None:
        sys.exit("I'm sorry, I could not find a suitable clang compiler.")
    if isinstance(compiler, ClangCompiler) and compiler.version[0] < 8:
        sys.exit(f"I'm sorry, but the clang compiler {compiler} is too old.")

    compiler = config.common_flags(compiler).add_flag('-O3').add_flag(
        '-g').add_flag('-ftrivial-auto-var-init=pattern')
    orig_tests = tests
    tests = expand_glob(tests, ['tests/*', 'benchmarks/*'])
    if not tests:
        no_tests_error(orig_tests, ['tests', 'benchmarks'])

    return run_test(compiler=compiler,
                    runner=runner,
                    reporter=reporter,
                    name='test-uninit',
                    tests=tests,
                    config=config,
                    timeout=timeout)


def run_test_tsan(compiler: Compiler,
                  reporter: Reporter,
                  tests: List[str],
                  config: Config,
                  timeout: Optional[float] = None) -> bool:
    reporter.log('Running tests with thread sanitizer', 'title')

    reporter.log('Not implemented')
    return True

    runner = TsanRunner()
    compiler = config.find_compiler() if compiler is None else compiler
    if not isinstance(compiler, ClangCompiler):
        reporter.log(
            'Thread sanitizer is supported only with clang. Skipping test')
        return True

    compiler = common_flags(compiler).add_flag('-fsanitize=thread')
    orig_tests = tests
    tests = expand_glob(tests, ['tests/*'])
    if not tests:
        no_tests_error(orig_tests, ['tests'])

    return run_test(compiler=compiler,
                    runner=runner,
                    reporter=reporter,
                    name='test-tsan',
                    tests=tests,
                    timeout=timeout)


def run_benchmark_all(compiler: Optional[Compiler],
                      reporter: Reporter,
                      tests: List[str],
                      config: Config,
                      timeout: Optional[float] = None) -> bool:
    reporter.log('Running benchmark', 'title')
    runner = NvprofRunner() if config.nvprof else Runner()
    compiler = config.find_compiler() if compiler is None else compiler
    if compiler is None:
        raise RuntimeError("Could not find a suitable compiler")
    compiler = config.common_flags(compiler).add_flag('-O3').add_flag('-g')
    orig_tests = tests
    tests = expand_glob(tests, ['benchmarks/*'])
    if not tests:
        no_tests_error(orig_tests, ['benchmarks'])

    return run_benchmark(compiler=compiler,
                         runner=runner,
                         reporter=reporter,
                         name='benchmark-all',
                         tests=tests,
                         config=config,
                         timeout=timeout)


def compile_assembly(compiler: Compiler,
                     reporter: Reporter,
                     tests: List[str],
                     config: Config,
                     timeout: Optional[float] = None) -> bool:
    reporter.log('Compiling to assembly', 'title')
    compiler = config.find_compiler() if compiler is None else compiler
    if compiler is None:
        raise RuntimeError("Could not find a suitable compiler")
    compiler = config.common_flags(compiler).add_flag('-O3').add_flag(
        '-S').add_flag('-fverbose-asm')

    rep = reporter.analysis_group('assembly')
    output = rep.compilation(compiler.add_source(
        config.source)).compile(out_file=config.binary)
    if not output.is_success():
        return False

    assembly = open(config.binary, 'r').read()
    if len(assembly.encode('utf-8')) > MAX_ASSEMBLY_OUTPUT:
        assembly = "Generated assembly was too long and wasn't stored"
    rep.analyze(assembly)

    return True


def expand_glob(globs: List[str], default: List[str]) -> List[str]:
    if not globs:
        globs = default
    tests = []
    for pattern in globs:
        if os.path.exists(pattern):
            tests.append(pattern)
        else:
            tests.extend(sorted(glob.glob(pattern)))

    return tests


def no_tests_error(orig_tests: List[str], dirs: List[str]):
    if orig_tests:
        quote = '"'
        sys.exit(
            f'The specified tests {", ".join(quote + t + quote for t in orig_tests)} don\'t match any files. '
            + 'Please give a path to test file or a glob expanding to tests')
    else:
        assert len(dirs) > 0
        if len(dirs) == 1:
            its = 'its'
            dirs = dirs[0]
        else:
            its = 'their'
            dirs = ', '.join(dirs[:-1]) + ' or ' + dirs[-1]

        sys.exit(
            f'Couldn\'t find default tests. Have you accidentally deleted {dirs} folder, or {its} contents?'
        )
