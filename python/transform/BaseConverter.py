class BaseConverter():
    def __init__(self, no_save: bool = False):
        self.operands = dict()
        self.tensors = dict()
        self.shapes = dict()
        self.input_names = list()
        self.output_names = list()
        
    def generate_mlir(self, mlir_file: str=None):
        raise NotImplementedError('generate_mlir')
    
    def addOperand(self, name, op):
        if name in self.operands:
            if self.operands[name] != op:
                raise KeyError("operand {} conflict".format(name))
            return
        self.operands[name] = op

    def getOperand(self, name):
        if name not in self.operands:
            raise KeyError("operand {} not found".format(name))
        return self.operands[name]