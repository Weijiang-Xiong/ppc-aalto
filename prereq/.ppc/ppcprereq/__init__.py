from typing import List, Optional
from ppcgrader.compiler import Compiler, find_clang_compiler, find_gcc_compiler, find_nvcc_compiler
import ppcgrader.config
from os import path


class Config(ppcgrader.config.Config):
    def __init__(self):
        self.source = 'prereq.cc'
        self.binary = 'prereq'
        self.tester = path.join(path.dirname(__file__), 'tester.cc')
        self.gpu = False

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
            compiler = compiler.add_flag('-iquote', include_path)
        return compiler

    def find_compiler(self) -> Optional[Compiler]:
        return find_gcc_compiler() or find_clang_compiler()

    def parse_output(self, output):
        time = None
        errors = None
        input_data = {}
        output_data = {}
        output_errors = {}
        statistics = {}
        triples = []
        size = None
        for line in output.splitlines():
            what, *rest = line.split('\t')
            if what == 'result':
                errors = {'fail': True, 'pass': False, 'done': False}[rest[0]]
            elif what == 'time':
                time = float(rest[0])
            elif what == 'perf_wall_clock_ns':
                time = int(rest[0]) / 1e9
                statistics[what] = int(rest[0])
            elif what.startswith('perf_'):
                statistics[what] = int(rest[0])
            elif what in ['error_magnitude', 'threshold']:
                output_errors[what] = float(rest[0])
            elif what in ['ny', 'nx', 'y0', 'x0', 'y1', 'x1']:
                input_data[what] = int(rest[0])
            elif what == 'avg':
                output_data[what] = [float(x) for x in rest]
            elif what == 'size':
                size = rest[0]
            elif what == 'triple':
                triples.append([float(x) for x in rest])
        if size == "small":
            nx = input_data["nx"]
            ny = input_data["ny"]
            assert len(triples) == nx * ny
            input_data["data"] = [
                triples[i * nx:(i + 1) * nx] for i in range(ny)
            ]
        return time, errors, input_data, output_data, output_errors, statistics

    def explain_terminal(self, output, color=False):
        from .info import explain_terminal
        return explain_terminal(output, color)
