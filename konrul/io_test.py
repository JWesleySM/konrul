import contextlib
import enum
import io
import itertools
import math
import re

class TestingReturnCode(enum.Enum):
  """A enumeration to describe the possible results of testing a candidate.
   - SUCCESS: the candidate produces the expected output for all the input samples
   - TYPE_DISCARDED: the candidate was discarded because it does not type-check agains the IO
   - RUNTIEM_ERROR: there was a runtime error while executing the candidate
   - CANDIDATE_TRIED: the candidate was succesfully executed, but it does not produce the
                      correct output in some of the IO samples.
  """
  SUCCESS = 0
  TYPE_DISCARDED = 1
  RUNTIME_ERROR = 2
  CANDIDATE_TRIED = 3


class InsufficientElements(Exception):
  """An excepetion to indicate that there are no enough elements for a candidate in
  an IO sample.
  """
  pass


def _is_compatible(tensor_order, values):
  """Check if a tensor can be assigned to a list of values.
  The input value cannot be a single number if the tensor
  has order bigger than 0. Analogously, the value must be a single
  number if the tensor has order 0.
  """
  if tensor_order > 0 and len(values) == 1:
    return False
  if tensor_order == 0 and len(values) > 1:
    return False

  return True


def _is_io_compatible(c, io):
  """Check if the candidate is unsuitable given the shape of the io"""
  # Regarding output, its value cannot be a single number if the tensor
  # has order bigger than 0. Analogously the output value must be a single
  # number if the output tensor has order 0
  if not _is_compatible(c.lhs_tensor.order, io.output.values):
    return False

  # A candidate can be typed discarded given an IO sample if
  #  1. number of tensors with order bigger than 0 > number of inputs which are lists

  tensor_orders = [c.lhs_tensor.order] + [t.order for t in c.rhs_tensors]
  n_scalars_candidate = sum(1 for ord in tensor_orders[1:] if ord == 0)
  n_scalars_io = sum(1 for var in io.input.values() if len(var.values) == 1)
  if n_scalars_candidate > 0 and n_scalars_io == 0:
    return False
  
  #  2. number of tensors with order 0 > number of inputs which are a single integer.
  n_non_scalars_candidate = len(c.rhs_tensors) - n_scalars_candidate
  n_non_scalars_io = len(io.input) - n_scalars_io
  if n_non_scalars_candidate > 0 and n_non_scalars_io == 0:
    return False

  return True


def _is_valid_substitution(substitution, inputs):
  """This function checks if a substitution is valid, i.e., if the 
  mappings input variable -> tensor are type-valid.
  """
  bond = dict()
  for input_var, tensor in substitution:
    if not _is_compatible(tensor.order, inputs[input_var].values):
        return False
    
    # A same tensor cannot be bond to two different inputs.
    if tensor in bond:
      if bond[tensor.name] != input_var:
        return False
    bond[tensor.name] = input_var

  return True


def _get_substitutions_permutation(candidate, io_sample):
  """Return all the possible substitutions for a candidate program
  given an IO sample.
  """
  # We only need to bind input variables to unique references in the
  # program, hence, a set is used.
  tensors = set(t for t in candidate.rhs_tensors if not t.name.isnumeric())
  taco_input_perm = []
  input_list = io_sample.input
  for p in itertools.permutations(input_list.keys(), len(tensors)):
    input_combs = list(zip(p, tensors))
    if _is_valid_substitution(input_combs, input_list):      
      taco_input_perm.append(input_combs)

  return taco_input_perm 


def _build_environment(lhs, lhs_order, substitution, io):
  """Builds an enviroment for free variables to create a PyTaco program."""
  env = dict()
  env[lhs.name] = (1, io.output.dims, [0]) if lhs_order == 0 else (io.output.total_elem, io.output.dims, [0] * io.output.total_elem)
  for input_var, tensor in substitution:
    env[tensor.name] = (io.input[input_var].total_elem, io.input[input_var].dims, io.input[input_var].values)
  
  return env


def _write_pytaco_program(candidate, env):
  """Write a program using PyTACO, TACO's Python API."""
  # The tensors in PyTaco must be declared with fixed dimension lengths.
  # We determine how the elements will be distributed by computing the nth 
  # root of the number of elements, where 'n' is the order of the tensor.
  tensors = [candidate.lhs_tensor] + candidate.rhs_tensors
  defined = dict([(t.name, False) for t in tensors])
  # Import PyTaco and NumPy.
  imports = 'import pytaco as pt\nimport numpy as np\n'
  # Declare tensors.
  t_declarations = ''
  t_initializations = ''
  for t in tensors:
    if defined[t.name]:
      continue
    
    # Skip constants
    if t.name.isnumeric():
      continue
    
    order = t.order
    t_declarations += f'{t.name} = '
    if order == 0:
      t_declarations += f'pt.tensor({env[t.name][2][0]}, dtype = pt.int32)\n'
      
      defined[t] = True
      continue
    else:
      if env[t.name][1]:
        dims = env[t.name][1]
      else:
        elements_by_dimension = math.ceil(env[t.name][0] ** (1/order)) if order > 0 else 1
        if elements_by_dimension ** order > len(env[t.name][2]):
          raise InsufficientElements(f'Not enough elements for tensor {t} (needs {elements_by_dimension ** order} and there are only {len(env[t.name][1])} available)')
    
        dims = [elements_by_dimension] * order

      format = ['pt.dense'] * order
      format_as_str = str(format).translate({39 : None})
      t_declarations += f'pt.tensor({dims}, fmt = pt.format({format_as_str}), dtype = pt.int32, name = \'{t.name}\')\n'

    # Initialize non-scalar tensors.
    values = env[t.name][2]
    values_idx = 0
    if env[t.name][1]:
      dims = env[t.name][1]
      coords = []
      for i in range(len(dims)):
        coords.append([*(range(dims[i]))])
    else:
      coords = [[*(range(elements_by_dimension))] for _ in range(order)]
    for coord in itertools.product(*coords):
      t_initializations += f'{t.name}.insert({list(coord)}, {values[values_idx]})\n'
      values_idx += 1

    defined[t.name] = True

  # Write computation and evaluate the left-hand side.
  index_vars_definition = 'i, j, k, l, m, n = pt.get_index_vars(6)\n'

  computation = f'{candidate.lhs_tensor.name}[{candidate.lhs_tensor.indexing}] = ' if candidate.lhs_tensor.order > 0 else f'{candidate.lhs_tensor.name}[None] = '
  for i in range(candidate.n_tensors - 1):
    tensor = candidate.rhs_tensors[i]
    tensor_comp = '-' if tensor.negated else ''
    if tensor.order == 0:
      tensor_comp += tensor.name if tensor.name.isnumeric() else f'{tensor.name}[None]'
    else:
      tensor_comp += tensor.name + '[' + tensor.indexing + ']'
    computation += tensor_comp
    
    if i < candidate.n_arithops:
      computation += f' {candidate.arithops[i]} '
 
  computation += '\na.evaluate()\n'
  # Convert to a NumPy flatten array.
  conversion = 'flatten_a = a.to_array().flatten()\n'
  # Print out results.
  # Set NumPy print options so the array is not truncated when printed.
  print_results = 'np.set_printoptions(threshold=np.inf)\n'
  print_results += 'print(flatten_a)\n'

  pytaco_program = imports + t_declarations + t_initializations + index_vars_definition + computation + conversion + print_results
  return pytaco_program
  

def _test_as_pytaco(candidate, io_sample, substitution):
  """Check if a candidate is the correct solution by implementing and 
  interpreting a PyTaco program given a substituion.
  """
  try:
    env = _build_environment(candidate.lhs_tensor, candidate.lhs_tensor.order, substitution, io_sample)
    pytaco_program = _write_pytaco_program(candidate, env)

  except InsufficientElements as ie:
    raise RuntimeError('Invalid substitution' + ': ' + str(ie))

  # Get output from Python dynamically executed code
  # https://stackoverflow.com/a/3906390
  f = io.StringIO()
  with contextlib.redirect_stdout(f):
    exec(pytaco_program)

  taco_output = [int(value) for value in re.split('\[|\]|\n| ', f.getvalue()) if value.lstrip('-').isnumeric()]
  return taco_output 


def _test_substitution(substitution, c, io_set):
  """Check if a candidate is the solution using a specific substution."""
  try:
    # We first check against the first sample in the IO set
    taco_output = _test_as_pytaco(c, io_set[0], substitution)
    if taco_output == io_set[0].output.values:
      # A candidate is correct if it returns the correct output for all
      # the elements in the IO set.
      for io in io_set[1:]:
        taco_output = _test_as_pytaco(c, io, substitution)
        if taco_output != io.output.values:
          return False
        
      return True
    else:
      return False
  except RuntimeError as e:
    raise e
  

def io_test(candidate, io_set):
  """Check if a candidate is the solution for the synthesis problem."""
  # We can discard candidates based only in the shape of the IO.
  # Since all IO samples have the same shape, we need to check only one item 
  # from the IO set.
  if not _is_io_compatible(candidate, io_set[0]):
    print(f'Ruling out {candidate}')
    return TestingReturnCode.TYPE_DISCARDED, None
  
  print(f'Running {candidate}', end=" -> ")
  input_substitutions = _get_substitutions_permutation(candidate, io_set[0])
  n_runtime_errors = 0
  # We check a candidate with all the possible substitions. We stop
  # as soon as we find the first substitution that leads to the 
  # correct answer.
  for substitution in input_substitutions:
    try:
      if _test_substitution(substitution, candidate, io_set):
        return TestingReturnCode.SUCCESS, substitution
    except RuntimeError:
      n_runtime_errors += 1
      continue
  
  # If there was an runtime error for all the possible substitutions for this candidate
  # we classifiy it as RUNTIME_ERROR, otherwise at there was at least one valid
  # substitution, but still gives us the wrong output.
  if n_runtime_errors == len(input_substitutions):
    return TestingReturnCode.RUNTIME_ERROR, None
  else:
    return TestingReturnCode.CANDIDATE_TRIED, None