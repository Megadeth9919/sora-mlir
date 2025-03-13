from graph_ir import *
from inst import *
import numpy as np
from rtl_model import *

import os
import pytest

# 测试目录
PATH = os.path.dirname(os.path.abspath(__file__))


@pytest.mark.skip(reason="not implemented")
def test_lower_misc_transpose01():
    config_file = os.path.join(PATH, 'case', 'transpose01')
    # config_file = "/home/wangjiaqi/workspace/sora_cc/case/transpose01"
    rtlmodel = RTLModel(config_file_path=config_file, debug=True)

    rtlmodel.run()
    pass
