import json
from typing import Dict, List, NamedTuple


class Variable(NamedTuple):
  """ A Variable object holds its dimension (1 for scalars and the length in case of arrays)."""
  total_elem: int
  dims: List[int]
  values: List[int]


class IOExample(NamedTuple):
  """ An IOExample object contains an input dictionary mapping names to Variables.
  The key in this case is a symbolic  name. Finally, this object keeps the output
  variable. """
  input: Dict[str, Variable]
  output: Variable


"""IOHandler is responsible to read an IO file in the JSON format containing 
the different IO samples and convert it into a set of IOExample objects.
"""

def parse_io(io_path):
  io_set = []
  try:
    with open(io_path, 'r') as io_file:
      io_pairs = json.load(io_file)
  except FileNotFoundError as e:
    raise e
     
  for sample in io_pairs:
    io_input_vars = dict()
    output_values = list(sample['output'].values())[0]
    var_length = int(output_values[0])
    if len(output_values) == 3:
      dimensions = [int(value) for value in output_values[1]]
      values = [int(value) for value in output_values[2].split()]
    else:
      dimensions = []
      values = [int(value) for value in output_values[1].split()]
    io_output_var = Variable(var_length, dimensions, values)
    for in_var, in_values in sample['input'].items():
      var_length = int(in_values[0])
      if len(in_values) == 3:
        dimensions = [int(value) for value in in_values[1]]
        values = [int(value) for value in in_values[2].split()]
      else:
        dimensions = []
        values = [int(value) for value in in_values[1].split()]
      
      io_input_vars[in_var] = Variable(var_length, dimensions, values)
    io_set.append(IOExample(io_input_vars, io_output_var))
    
  return io_set