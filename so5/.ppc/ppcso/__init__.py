from typing import List, Optional
from ppcgrader.compiler import Compiler, find_clang_compiler, find_gcc_compiler, find_nvcc_compiler
import ppcgrader.config
from os import path
from operator import methodcaller


class Config(ppcgrader.config.Config):
    def __init__(self, openmp: bool, gpu: bool = False):
        self.source = 'so.cu' if gpu else 'so.cc'
        self.binary = 'so'
        self.tester = path.join(path.dirname(__file__), 'tester.cc')
        self.gpu = gpu
        self.openmp = openmp and not gpu

    def test_command(self, test: str) -> List[str]:
        return [path.join('./', self.binary), f'--test', test]

    def benchmark_command(self, test: str) -> List[str]:
        return [path.join('./', self.binary), test]

    def common_flags(self, compiler: Compiler) -> Compiler:
        include_paths = [
            path.join(path.dirname(__file__), 'include'),
            path.normpath(
                path.join(path.dirname(__file__), '../ppcgrader/include'))
        ]
        for include_path in include_paths:
            if self.gpu:
                compiler = compiler.add_flag('-I', include_path)
            else:
                compiler = compiler.add_flag('-iquote', include_path)
        if self.openmp:
            compiler = compiler.add_omp_flags()
        return compiler

    def find_compiler(self) -> Optional[Compiler]:
        if self.gpu:
            return find_nvcc_compiler()
        else:
            return find_gcc_compiler() or find_clang_compiler()

    def parse_output(self, output):
        input_data = {
            "n": None,
            "data": None,
        }
        output_data = {
            "result": None,
        }
        output_errors = {
            "type": 0,
            "correct": None,
        }

        statistics = {}

        def parse_matrix(string):
            M = string.strip('[]').split(';')
            M = [row.strip() for row in M]
            M = [row.split(" ") for row in M]
            M = [[int(e) for e in row if e != ''] for row in M]
            return M

        for line in output.splitlines():
            splitted = line.split('\t')
            if splitted[0] == 'result':
                errors = {
                    'fail': True,
                    'pass': False,
                    'done': False
                }[splitted[1]]
            elif splitted[0] == 'time':
                time = float(splitted[1])
            elif splitted[0] == 'perf_wall_clock_ns':
                time = int(splitted[1]) / 1e9
                statistics[splitted[0]] = int(splitted[1])
            elif splitted[0].startswith('perf_'):
                statistics[splitted[0]] = int(splitted[1])
            elif splitted[0] == 'n':
                input_data['n'] = int(splitted[1])
            elif splitted[0] == 'input':
                input_data["data"] = parse_matrix(splitted[1])[0]
            elif splitted[0] == 'output':
                output_data["result"] = parse_matrix(splitted[1])[0]
            elif splitted[0] == 'correct':
                output_errors["correct"] = parse_matrix(splitted[1])[0]
            elif splitted[0] == 'error_type':
                output_errors["type"] = int(splitted[1])

        return time, errors, input_data, output_data, output_errors, statistics

    # TODO: Implement method
    def format_output(self, output) -> Optional[str]:
        raise NotImplementedError

    def explain_terminal(self, output, color=False) -> Optional[str]:
        from .info import explain_terminal
        return explain_terminal(output, color)
