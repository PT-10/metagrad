import metagrad as mg
import numpy as np
from graphviz import Digraph

def tensor_to_string(tensor):
    """Converts tensor data to a string representation."""
    if tensor is not None:
        if tensor.ndim > 6 or tensor.size > 6:  # Adjust based on preference
            return f"shape {tensor.shape}"
        else:
            return np.array2string(tensor, formatter={'float_kind': lambda x: f"{x:.4f}"})
    else:
        return "None"

def trace(root):
    nodes, edges = set(), set()
    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)
    build(root)
    return nodes, edges

def draw_dot(root, format='svg', rankdir='LR'):
    """
    format: png | svg | ...
    rankdir: TB (top to bottom graph) | LR (left to right)
    """
    assert rankdir in ['LR', 'TB']
    nodes, edges = trace(root)
    dot = Digraph(format=format, graph_attr={'rankdir': rankdir})
    
    for n in nodes:
        data_str = tensor_to_string(n.data)
        grad_str = tensor_to_string(n.grad)
        label = f"{{ data {data_str} | grad {grad_str} }}"
        dot.node(name=str(id(n)), label=label, shape='record')
        if hasattr(n, '_opt') and n._opt:
            opt_node_name = str(id(n)) + n._opt
            dot.node(name=opt_node_name, label=n._opt)
            dot.edge(opt_node_name, str(id(n)))
    
    for n1, n2 in edges:
        edge_start = str(id(n1))
        edge_end = str(id(n2))
        if hasattr(n2, '_opt') and n2._opt:
            edge_end += n2._opt
        dot.edge(edge_start, edge_end)
    dot.render('graph/diagram')
    return dot

# Example usage
tensor_a = mg.Tensor(np.array([1,2,1]))
tensor_b = ([3,1,1])
tensor_c = (([5,6,1]))
tensor_d = mg.Tensor(([9,8,1]))
out1 = tensor_a - tensor_b
new = ([[1, 2, 3], [4, 5, 6]])
out2 = out1/new
out3 = tensor_c + tensor_d
outmid = out2+out2+out2
outout = out3 + outmid
out = outout*(3)
print("Out", out)
out.backward()  
graph = draw_dot(out)
