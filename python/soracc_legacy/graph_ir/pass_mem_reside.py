from .graph_ir import *
import json
def pass_mem_reside(g: StaticGraph):
    for op in g.get_ops() :
        if isinstance(op, View):
            op.get_outputs()[0].reside_ddr = op.get_inputs()[0].reside_ddr
            continue
        
        inputs = op.get_inputs()
        outputs = op.get_outputs()
        
        hbm_flag = False
        for input in inputs:
            if not input.get_reside_DDR() :
                hbm_flag = True
            
        for output in outputs:
            if(hbm_flag or output.cached or output.const or output in g.get_outputs() or output.data_type == DataType.float32):
                output.set_reside_DDR()
            else :
                output.set_reside_HBM()
                    
