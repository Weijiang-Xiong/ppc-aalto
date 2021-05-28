from typing import List, Optional
from ppcgrader.compiler import Compiler, find_clang_compiler, find_gcc_compiler, find_nvcc_compiler
import ppcgrader.config
from os import path
from operator import methodcaller


class Config(ppcgrader.config.Config):
    def __init__(self, openmp: bool = False, gpu: bool = False):
        self.source = 'is.cu' if gpu else 'is.cc'
        self.binary = 'is'
        self.tester = path.join(path.dirname(__file__), 'tester.cc')
        self.gpu = gpu
        self.openmp = openmp and not gpu

    def test_command(self, test: str) -> List[str]:
        return [path.join('./', self.binary), '--test', test]

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
        time = None
        errors = None
        input_data = {"nx": None, "ny": None}
        output_data = {
            "y0": None,
            "y1": None,
            "x0": None,
            "x1": None,
            "outer": None,
            "inner": None
        }
        output_errors = {
            "expected": {
                "y0": None,
                "y1": None,
                "x0": None,
                "x1": None,
                "outer": None,
                "inner": None
            }
        }
        statistics = {}
        size = None
        target = None
        triples = []
        for line in output.splitlines():
            what, arg = line.split('\t')
            if what == 'result':
                errors = {'fail': True, 'pass': False, 'done': False}[arg]
            elif what == 'time':
                time = float(arg)
            elif what == 'perf_wall_clock_ns':
                time = int(arg) / 1e9
                statistics[what] = int(arg)
            elif what.startswith('perf_'):
                statistics[what] = int(arg)
            elif what in ['ny', 'nx']:
                input_data[what] = int(arg)
            elif what in ['error_magnitude', 'threshold']:
                output_errors[what] = float(arg)
            elif what == 'what':
                target = {
                    'expected': output_errors['expected'],
                    'got': output_data,
                }[arg]
            elif what in ['y0', 'y1', 'x0', 'x1']:
                target[what] = int(arg)
            elif what in ['inner', 'outer']:
                parsed = [float(c) for c in arg.split(',')]
                target[what] = parsed
            elif what == 'size':
                size = arg
            elif what == 'triple':
                parsed = [float(c) for c in arg.split(',')]
                triples.append(parsed)
        if size == "small":
            nx = input_data["nx"]
            ny = input_data["ny"]
            assert len(triples) == nx * ny
            input_data["data"] = [
                triples[i * nx:(i + 1) * nx] for i in range(ny)
            ]
        if errors:
            output_errors["wrong_output"] = True

        return time, errors, input_data, output_data, output_errors, statistics

    def explain_terminal(self, output, color=False) -> Optional[str]:
        from .info import explain_terminal
        return explain_terminal(output, color)
