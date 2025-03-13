

class HookInfo:
    __slots__ = ('op_name', 'offset', 'size', 'input', 'output')
    def __init__(self) -> None:
        self.op_name: str = ""
        self.offset: int = 0
        self.size: int = 0
        self.input: dict[str, list[int]] = {}
        self.output: dict[str, list[int]] = {}
    

    def to_dict(self):
        return {
            'op_name': self.op_name,
            'offset': self.offset,
            'size': self.size,
            'input': self.input,
            'output': self.output
        }

class InfoCollector:
    def __init__(self):
        self._hook: list[HookInfo] = []

    def add(self, hook: HookInfo):
        self._hook.append(hook)

    def get_insts(self):
        return self._hook
    
    def to_dict(self):
        return {
            '_hook': [hook.to_dict() for hook in self._hook]
        }
 
    def __getitem__(self, index: int):
        return self._hook[index]
    
    def __len__(self):
        return len(self._hook)
