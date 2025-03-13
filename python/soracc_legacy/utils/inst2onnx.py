import argparse
import onnx
import json
from onnx import helper


class Dag:
    def __init__(self):
        self.adjacency_list = {}
        self.inst_nodes = []
        
    def add_vertex(self, vertex, inst):
        if vertex not in self.adjacency_list:
            self.adjacency_list[vertex] = []
            self.inst_nodes.append(inst)

    def add_edge(self, start_vertex, end_vertex):
        """添加有向边，从start_vertex指向end_vertex"""
        if start_vertex in self.adjacency_list and end_vertex in self.adjacency_list:
            self.adjacency_list[start_vertex].append(end_vertex)
        else:
            raise ValueError("Both vertices must be present in the graph before adding a directed edge.")

    def remove_edge(self, start_vertex, end_vertex):
        """移除有向边，从start_vertex指向end_vertex"""
        if start_vertex in self.adjacency_list and end_vertex in self.adjacency_list:
            if end_vertex in self.adjacency_list[start_vertex]:
                self.adjacency_list[start_vertex].remove(end_vertex)

    def remove_vertex(self, vertex):
        """移除一个顶点及其所有相关边"""
        if vertex in self.adjacency_list:
            # 移除所有指向该顶点的边
            for neighbor in self.adjacency_list:
                if vertex in self.adjacency_list[neighbor]:
                    self.adjacency_list[neighbor].remove(vertex)
            # 移除该顶点及其出边
            del self.adjacency_list[vertex]

    def display(self):
        """显示图的邻接表"""
        for vertex, neighbors in self.adjacency_list.items():
            print(f"{vertex}: {neighbors}")

    def is_reachable(self, start, target):
        """判断从start节点是否可以到达target节点"""
        if start not in self.adjacency_list or target not in self.adjacency_list:
            return False

        visited = set() 
        queue = [start]  
        visited.add(start)

        while queue:
            current = queue.pop(0)  
            if current == target:
                return True  

            for neighbor in self.adjacency_list[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

        return False  
        
class OnnxConverter:  
  @staticmethod
  def onnx_make_nodes(dag: Dag):
    nodes = []
    inputs_list = [[] for _ in range(len(dag.adjacency_list))]
    outputs_list = [[] for _ in range(len(dag.adjacency_list))]
    attrs_list = [{} for _ in range(len(dag.adjacency_list))]
    optype_list = ['' for _ in range(len(dag.adjacency_list))]
    edge_id = 0
    for idx, inst in enumerate(dag.inst_nodes):
      
      for next_idx in dag.adjacency_list[idx]:
        outputs_list[idx].append(str(edge_id))
        inputs_list[next_idx].append(str(edge_id))
        edge_id += 1
        
      attrs = inst
      if inst['inst_type'] == 'misc_inst':
        optype_list[idx] = attrs['op']
        del attrs['op']
      elif inst['inst_type'] == 'load' or inst['inst_type'] == 'store':
        optype_list[idx] = attrs['inst_type'] + ' ' + attrs['mode']
        del attrs['mode']
      else:
        optype_list[idx] = attrs['inst_type']
        
      del attrs['wait']
      del attrs['release']
      del attrs['inst_type']
      attrs_list[idx] = attrs

    for idx, (optype, inputs, outputs, attrs) in enumerate(zip(optype_list, inputs_list, outputs_list, attrs_list)):
      if idx == 0:
        inputs = ['start']
      elif idx == len(inputs_list) - 1:
        outputs = ['end']
      node = helper.make_node(
        optype,  # 节点的操作类型，如"Conv"、"Relu"、"MatMul"等
        inputs,  # 输入张量的名称列表
        outputs,  # 输出张量的名称列表
        name=None,  # 节点的名称（可选）
        **attrs  # 节点的属性字典（可选）
      )
      nodes.append(node)
    return nodes

  @staticmethod
  def onnx_make_graph(nodes):
    graph = helper.make_graph(
      nodes,
      'inst_graph',  # 图的名称
      [helper.make_tensor_value_info('start', onnx.TensorProto.FLOAT, [])],  # 输入张量信息
      [helper.make_tensor_value_info('end', onnx.TensorProto.FLOAT, [])]  # 输出张量信息
    )
    return graph

def load_json_file(file_path):
    """从文件中加载JSON数据"""
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def convert_inst_type(inst_type):
  if inst_type == 'mm_inst':
    return 'MM'
  elif inst_type == 'load':
    return 'LD'
  elif inst_type == 'store':
    return 'ST'
  elif inst_type == 'misc_inst':
    return 'MISC'
  elif inst_type == 'sys':
    return 'SYS'
  elif inst_type == 'rs_inst':
    return 'RS'
  

def create_dag(instructions):
  dag = Dag()
    
  # 创建节点
  for idx, inst in enumerate(instructions):
      dag.add_vertex(idx, inst)
      
  # 添加依赖关系的边
  for idx, inst in enumerate(instructions):
      if 'wait' in inst and inst['wait']:
          for dep in inst['wait']:
              # 找到依赖的指令
              for dep_idx, dep_inst in enumerate(instructions[:idx]):
                  if dep == convert_inst_type(dep_inst['inst_type']) and convert_inst_type(inst['inst_type']) in dep_inst['release']:
                    dag.add_edge(dep_idx, idx)
                    dep_inst['release'].remove(convert_inst_type(inst['inst_type']))
                    break
  for idx, inst in enumerate(instructions):
    for pre_idx, pre_inst in enumerate(instructions[:idx][::-1]):
      if inst['inst_type'] == pre_inst['inst_type'] and not dag.is_reachable(idx - pre_idx - 1, idx):
        dag.add_edge(idx - pre_idx - 1, idx)
        break
      
  return dag

def inst2onnx(file_path,index:int=0):
  instructions = load_json_file(file_path)
  dag = create_dag(instructions)
  # dag.display()
  nodes = OnnxConverter.onnx_make_nodes(dag)
  graph = OnnxConverter.onnx_make_graph(nodes)
  model = helper.make_model(graph)
  onnx.save(model, f"{file_path}_{index}.onnx")
  
  
if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('file_path', type=str, help='inst.json file path')
  args = parser.parse_args()
  
  # convert
  inst2onnx(args.file_path)
  
