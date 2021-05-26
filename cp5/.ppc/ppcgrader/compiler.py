import subprocess
from typing import List
import re
import copy
from ppcgrader.logging import log_command
import platform
import sys

MAX_COMPILER_OUTPUT = 30000  # Characters
MIN_GCC, MAX_GCC = 8, 15
MIN_CLANG, MAX_CLANG = 6, 20

GCC_BINARIES = ['g++'] + [f"g++-{v}" for v in range(MIN_GCC, MAX_GCC + 1)]
CLANG_BINARIES = ['clang++'] + \
    [f"clang++-{v}" for v in range(MIN_CLANG, MAX_CLANG + 1)]
NVCC_BINARIES = ['nvcc']


class CompilerOutput:
    def __init__(self, stdout: str, stderr: str, returncode: int):
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode

    def is_success(self) -> bool:
        return self.returncode == 0


class Compiler:
    def __init__(self, program: str, common_flags: List[str]):
        self.sources = []
        self.flags = []
        self.program = program
        self.common_flags = common_flags

    def add_source(self, file: str) -> 'Compiler':
        me = copy.deepcopy(self)
        me.sources.append(file)
        return me

    def add_flag(self, *flags: str) -> 'Compiler':
        me = copy.deepcopy(self)
        me.flags.extend(flags)
        return me

    def add_omp_flags(self) -> 'Compiler':
        self = self.add_flag('-fopenmp')
        return self

    def compile(self, out_file: str = 'a.out') -> CompilerOutput:
        try:
            args = [self.program
                    ] + self.common_flags + self.flags + self.sources + [
                        '-o', out_file
                    ]
            log_command(args)
            result = subprocess.run(args,
                                    timeout=10,
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE,
                                    encoding='utf-8',
                                    errors='utf-8')
        except subprocess.TimeoutExpired:
            return CompilerOutput(
                '',
                'Compilation process took too much time, killed the process',
                -1)

        return CompilerOutput(result.stdout[:MAX_COMPILER_OUTPUT],
                              result.stderr[:MAX_COMPILER_OUTPUT],
                              result.returncode)

    def is_valid(self) -> bool:
        try:
            args = [self.program, '-dM', '-fopenmp', '-E', '-']
            log_command(args, 2)
            return subprocess.run(args,
                                  input='',
                                  timeout=10,
                                  stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE,
                                  encoding='utf-8',
                                  errors='utf-8').returncode == 0
        except AssertionError:
            return False
        except subprocess.TimeoutExpired:
            sys.exit(
                'Testing compiler took too much time, killed the process.')

    def __str__(self):
        return self.program


class GccCompiler(Compiler):
    def __init__(self, program: str = 'g++'):
        super().__init__(program=program,
                         common_flags=[
                             '-std=c++2a',
                             '-Wall',
                             '-Wextra',
                             '-Werror',
                             '-Wno-error=unknown-pragmas',
                             '-Wno-error=unused-but-set-variable',
                             '-Wno-error=unused-local-typedefs',
                             '-Wno-error=unused-function',
                             '-Wno-error=unused-label',
                             '-Wno-error=unused-value',
                             '-Wno-error=unused-variable',
                             '-Wno-error=unused-parameter',
                             '-Wno-error=unused-but-set-parameter',
                             '-Wno-psabi',
                             '-march=native',
                             '-fdiagnostics-color=never',
                         ])

    def __repr__(self):
        return f'GCC compiler ({self.program})'


class ClangCompiler(Compiler):
    def __init__(self, program: str = 'clang++'):
        self.version, self.apple = ClangCompiler.__get_version(program)

        flags = [
            '-std=c++2a',
            '-Wall',
            '-Wextra',
            '-Werror',
            '-Wno-error=unknown-pragmas',
            # '-Wno-error=unused-but-set-variable',
            '-Wno-error=unused-local-typedefs',
            '-Wno-error=unused-function',
            '-Wno-error=unused-label',
            '-Wno-error=unused-value',
            '-Wno-error=unused-variable',
            '-Wno-error=unused-parameter',
            # '-Wno-error=unused-but-set-parameter',
            '-march=native',
        ]
        if platform.system() == 'Darwin' and platform.machine() == 'arm64':
            flags = flags[:-1]

        if self.version and self.version[0] >= 11 and self.apple == False:
            flags.append('-Wno-psabi')

        super().__init__(program=program, common_flags=flags)

    @staticmethod
    def __get_version(program: str):
        try:
            args = [program, '-dM', '-fopenmp', '-E', '-']
            if platform.system() == 'Darwin':
                args = [
                    program, '-dM', '-Xpreprocessor', '-fopenmp', '-E', '-'
                ]
            log_command(args, 2)
            output = subprocess.run(args,
                                    input='',
                                    timeout=10,
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE,
                                    encoding='utf-8',
                                    errors='utf-8')
            if output.returncode == 0:
                lines = output.stdout.split('\n')
                matches = [
                    re.match('^#define (\\w+) (.*)$', line) for line in lines
                ]
                macros = {
                    match.group(1): match.group(2)
                    for match in matches if match is not None
                }
                try:
                    apple = True if 'Apple' in macros['__VERSION__'] else False
                except:
                    apple = None
                if '__clang__' in macros and '_OPENMP' in macros:
                    try:
                        return [(int(macros['__clang_major__']),
                                 int(macros['__clang_minor__']),
                                 int(macros['__clang_patchlevel__'])), apple]
                    except TypeError:
                        pass

        except FileNotFoundError:
            pass

        return [None, None]

    def __repr__(self):
        return f'Clang compiler ({self.program})'

    def add_omp_flags(self) -> 'Compiler':
        # Apple clang doesn't have openmp compiled so we want to include the homebrew package.
        if platform.system() == 'Darwin':
            try:
                brew_dir_command = ['brew', '--prefix']
                log_command(brew_dir_command)
                brew_dir = subprocess.run(brew_dir_command,
                                          timeout=10,
                                          stdout=subprocess.PIPE,
                                          stderr=subprocess.PIPE,
                                          encoding='utf-8').stdout.strip()
            except FileNotFoundError:
                message = '''It seems that you don't have Homebrew installed.
It is needed if you want to run OpenMP tasks on macOS.
It would be great if you could install Homebrew from:

    https://brew.sh/

After that you need to install libomp with this command:

    brew install libomp

'''
                if platform.machine() != 'arm64':
                    message += '''
Or alternatively you can try to install GCC and use it instead of Clang:

    brew install gcc

'''
                message += '''Once you have done that, please try to run the same command again,
I should be able to find the right packages and compilers then!'''
                sys.exit(message)

            try:
                brew_libomp_command = ['brew', 'list', 'libomp']
                log_command(brew_libomp_command)
                assert (subprocess.run(brew_libomp_command,
                                       timeout=10,
                                       stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE).returncode == 0)
            except subprocess.TimeoutExpired:
                print('Could not check required packages. Continuing...')
            except AssertionError:
                message = '''It seems that you have got Homebrew installed, which is great!
However, it seems you do not have the libomp package installed.
This is needed if you want to use OpenMP with the Clang C++ compiler.

Could you please try to install libomp with this command:

    brew install libomp

'''
                if platform.machine() != 'arm64':
                    message += '''
Or alternatively you can try to install GCC and use it instead of Clang:

    brew install gcc

'''
                message += '''Once you have done that, please try to run the same command again,
I should be able to find the right packages and compilers then!'''
                sys.exit(message)

            self = self.add_flag('-Xpreprocessor', '-fopenmp')
            self = self.add_flag('-I', f'{brew_dir}/include')
            if sys.argv[1] != 'assembly':
                self = self.add_flag('-lomp')
                self = self.add_flag('-L', f'{brew_dir}/lib')

        else:
            self = self.add_flag('-fopenmp')
        return self

    def is_valid(self) -> bool:
        try:
            args = [self.program, '-dM', '-fopenmp', '-E', '-']
            if platform.system() == 'Darwin':
                args = [
                    self.program, '-dM', '-Xpreprocessor', '-fopenmp', '-E',
                    '-'
                ]
            log_command(args, 2)
            return subprocess.run(args,
                                  input='',
                                  timeout=10,
                                  stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE,
                                  encoding='utf-8',
                                  errors='utf-8').returncode == 0
        except AssertionError:
            return False
        except subprocess.TimeoutExpired:
            sys.exit(
                'Testing compiler took too much time, killed the process.')


class NvccCompiler(Compiler):
    def __init__(self, program: str = 'nvcc'):
        super().__init__(program=program,
                         common_flags=[
                             '-Xcompiler',
                             '-std=c++14',
                             '-Xcompiler',
                             '"-Wall"',
                             '-Xcompiler',
                             '"-Wextra"',
                             '-Xcompiler',
                             '"-Werror"',
                             '-Xcompiler',
                             '"-Wno-error=unknown-pragmas"',
                             '-Xcompiler',
                             '"-Wno-error=unused-local-typedefs"',
                             '-Xcompiler',
                             '"-Wno-error=unused-function"',
                             '-Xcompiler',
                             '"-Wno-error=unused-label"',
                             '-Xcompiler',
                             '"-Wno-error=unused-value"',
                             '-Xcompiler',
                             '"-Wno-error=unused-variable"',
                             '-Xcompiler',
                             '"-Wno-error=unused-parameter"',
                             '-Xcompiler',
                             '"-Wno-psabi"',
                             '-Xcompiler',
                             '"-march=native"',
                         ])

    def __repr__(self):
        return f'NVCC compiler ({self.program})'

    def compile(self, out_file: str = 'a.out') -> CompilerOutput:
        try:
            args = [self.program
                    ] + self.common_flags + self.flags + self.sources + [
                        '-o', out_file
                    ]
            log_command(args)
            result = subprocess.run(args,
                                    timeout=10,
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE,
                                    encoding='utf-8',
                                    errors='utf-8')
        except subprocess.TimeoutExpired:
            return CompilerOutput(
                '',
                'Compilation process took too much time, killed the process',
                -1)

        return CompilerOutput(result.stdout[:MAX_COMPILER_OUTPUT],
                              result.stderr[:MAX_COMPILER_OUTPUT],
                              result.returncode)

    def add_omp_flags(self) -> 'Compiler':
        self.common_flags += ['-Xcompiler' '-fopenmp']
        return self

    def is_valid(self) -> bool:
        try:
            args = [self.program, '-Xcompiler', '-dM', '-x', 'cu', '-E', '-']
            log_command(args, 2)
            return subprocess.run(args,
                                  input='',
                                  timeout=10,
                                  stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE,
                                  encoding='utf-8',
                                  errors='utf-8').returncode == 0
        except AssertionError:
            return False
        except subprocess.TimeoutExpired:
            sys.exit(
                'Testing compiler took too much time, killed the process.')


def find_gcc_compiler():
    best = None
    for program in GCC_BINARIES:
        try:
            # currently the arm version of gcc from homebrew doesn't come with libasan and libubsan in which case we don't want to choose gcc
            if platform.system() == 'Darwin' and platform.machine() == 'arm64':
                args = [
                    program, '-xc++', '-fsanitize=address',
                    '-fsanitize=undefined', '-', '-o', '/dev/null'
                ]
                log_command(args, 2)
                assert (subprocess.run(args,
                                       timeout=10,
                                       input="int main() { } ",
                                       text=True,
                                       stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE).returncode == 0)

            args = [program, '-dM', '-fopenmp', '-E', '-']
            log_command(args, 2)
            output = subprocess.run(args,
                                    input='',
                                    timeout=10,
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE,
                                    encoding='utf-8',
                                    errors='utf-8')
            if output.returncode == 0:
                lines = output.stdout.split('\n')
                matches = [
                    re.match('^#define (\\w+) (.*)$', line) for line in lines
                ]
                macros = {
                    match.group(1): match.group(2)
                    for match in matches if match is not None
                }

                if '__clang__' not in macros and '_OPENMP' in macros:
                    # We don't want to choose clang in any case
                    try:
                        version = (int(macros['__GNUC__']),
                                   int(macros['__GNUC_MINOR__']),
                                   int(macros['__GNUC_PATCHLEVEL__']))
                        if version[0] >= MIN_GCC and (best is None
                                                      or best[1] < version):
                            best = (program, version)
                    except TypeError:
                        pass

        except FileNotFoundError:
            pass
        except AssertionError:
            pass
    if best is not None:
        return GccCompiler(best[0])


def find_clang_compiler():
    best = None
    for program in CLANG_BINARIES + GCC_BINARIES:
        compiler = ClangCompiler(program)

        if compiler.version and compiler.version[0] >= MIN_CLANG and (
                best is None or best.version < compiler.version):
            best = compiler

    if best is not None:
        return best


def find_nvcc_compiler():
    best = None
    for program in NVCC_BINARIES:
        try:
            log_command([program], 2)
            output = subprocess.run([program],
                                    input='',
                                    timeout=10,
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE,
                                    encoding='utf-8',
                                    errors='utf-8')
            if output.returncode == 1:
                best = (program, 0)
        except FileNotFoundError:
            pass
    if best is not None:
        return NvccCompiler(best[0])
