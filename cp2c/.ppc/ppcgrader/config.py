from ppcgrader.compiler import Compiler
from ppcgrader.runner import RunnerOutput
from typing import Dict, List, Optional, Tuple


class Config:
    def __init__(self):
        self.source: str
        self.binary: str
        self.tester: str
        self.gpu: bool
        self.ignore_errors: bool
        raise NotImplementedError

    def test_command(self, test: str) -> List[str]:
        raise NotImplementedError

    def benchmark_command(self, test: str) -> List[str]:
        raise NotImplementedError

    def common_flags(self, compiler: Compiler) -> Compiler:
        raise NotImplementedError

    def find_compiler(self) -> Optional[Compiler]:
        raise NotImplementedError

    def parse_output(self,
                     output: str) -> Tuple[float, bool, Dict, Dict, Dict]:
        raise NotImplementedError

    def format_output(self, output: RunnerOutput) -> Optional[str]:
        raise NotImplementedError
