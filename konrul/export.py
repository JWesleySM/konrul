
def _is_pluseq_einsum(einsum_prog):
  # Optimization:
  # a(idx1) = b(idx1) + (sequence of multiplications) OR
  # a(idx1) = (sequence of multiplications) + b(idx1) ->
  # a += einsum_program
  tensor_on_both_sides_idxs = [i for i in range(einsum_prog.n_tensors - 1) if einsum_prog.rhs_tensors[i].indexing == einsum_prog.lhs_tensor.indexing]
  if len(tensor_on_both_sides_idxs) != 1:
    return False, -1
  
  tensor_on_both_sides_idx = tensor_on_both_sides_idxs[0]

  if tensor_on_both_sides_idx == 0:
    if einsum_prog.arithops[0] != '+':
      return False, -1

    rhs_arithops = einsum_prog.arithops[1:]
  elif tensor_on_both_sides_idx == einsum_prog.n_tensors - 2:
    if einsum_prog.arithops[einsum_prog.n_arithops - 1] != '+':
      return False, -1

    rhs_arithops = einsum_prog.arithops[:-1]
  else:
    return False, -1
  
  if not rhs_arithops:
    return False, -1
  
  if any(arithop != '*' for arithop in rhs_arithops):
    return False, -1
  
  return True, tensor_on_both_sides_idx

  
def _is_pure_einsum(einsum_prog):
  if einsum_prog.n_tensors == 3:
    if einsum_prog.rhs_tensors[0].name.isnumeric():
      return False
    
    if einsum_prog.lhs_tensor.order > einsum_prog.rhs_tensors[0].order:
      return False
    
    return True
  else:
    if all(arithop == '*' for arithop in einsum_prog.arithops):
      return True

  return False


def _pytorch_constant_codegen(constant_value):
  return f'cons = torch.tensor([{constant_value}])\n  '


def _einsum_indexing_codegen(indexing_expr):
  return indexing_expr.replace(',','') if indexing_expr else 'i'
  

def _pytorch_einsum_codegen(lhs_tensor, einsum_tensors):
  einsum_exp = '\'' + _einsum_indexing_codegen(einsum_tensors[0].indexing)
  einsum_args = '[-' if einsum_tensors[0].negated else '['
  einsum_args += einsum_tensors[0].name
  for t in einsum_tensors[1:]:
    einsum_exp += ',' + _einsum_indexing_codegen(t.indexing)
    t_name = '-' + t.name if t.negated else t.name
    einsum_args += ', ' + t_name
  
  einsum_exp += '->' + _einsum_indexing_codegen(lhs_tensor.indexing) + '\''
  return einsum_exp, einsum_args


def _pytorch_pure_einsum_codegen(einsum_prog):
  einsum_call = 'torch.einsum('
  einsum_exp, einsum_args = _pytorch_einsum_codegen(einsum_prog.lhs_tensor, einsum_prog.rhs_tensors)
  return einsum_prog.lhs_tensor.name + ' = ' + einsum_call + einsum_exp + ', ' + einsum_args + '])'


def _pytorch_pluseq_einsum_codegen(lhs_tensor, einsum_tensors):
  einsum_call = 'torch.einsum('
  einsum_exp, einsum_args = _pytorch_einsum_codegen(lhs_tensor, einsum_tensors)
  return lhs_tensor.name + ' += ' + einsum_call + einsum_exp + ',' + einsum_args + '])'


def _pytorch_extended_einsum_codegen(einsum_prog):
  constants = ''

  if einsum_prog.n_tensors == 1:
    tensor = einsum_prog.rhs_tensors[0]
    if tensor.name.isnumeric():
      constant_value = '-' + tensor.name if tensor.negated else tensor.name
      constants += _pytorch_constant_codegen(constant_value)
      torch_prog = einsum_prog.lhs_tensor.name + ' = cons '
    else:
      tensor_name = '-' + tensor.name if tensor.negated else tensor.name
      torch_prog = einsum_prog.lhs_tensor.name + ' = ' + tensor_name
  else:
    torch_prog = einsum_prog.lhs_tensor.name + ' = '
    for i in range(einsum_prog.n_arithops):
      tensor = einsum_prog.rhs_tensors[i]
      if tensor.name.isnumeric():
        constant_value = '-' + tensor.name if tensor.negated else tensor.name
        constants += _pytorch_constant_codegen(constant_value)
        torch_prog += 'cons'
      else:
        tensor_name = '-' + tensor.name if tensor.negated else tensor.name
        torch_prog += tensor_name

      torch_prog +=  ' ' + einsum_prog.arithops[i] + ' '
    
    tensor = einsum_prog.rhs_tensors[-1]
    if tensor.name.isnumeric():
      constants += _pytorch_constant_codegen(tensor.name)
      torch_prog += 'cons'
    else:
      tensor_name = '-' + tensor.name if tensor.negated else tensor.name
      torch_prog += tensor_name
  
  return constants + torch_prog


def pytorch_gen(einsum_prog):
  if _is_pure_einsum(einsum_prog):
    pytorch_prog = _pytorch_pure_einsum_codegen(einsum_prog)
  else:
    pluseq_einsum, pluseq_tensor_idx = _is_pluseq_einsum(einsum_prog)
    if pluseq_einsum:
      einsum_tensors = einsum_prog.rhs_tensors[1:] if pluseq_tensor_idx == 0 else einsum_prog.rhs_tensors[:-1]
      pytorch_prog = _pytorch_pluseq_einsum_codegen(einsum_prog.lhs_tensor, einsum_tensors)
    else:
      pytorch_prog = _pytorch_extended_einsum_codegen(einsum_prog)
  return pytorch_prog


def taco_gen(einsum_prog):
  def gen_taco_tensor(einsum_tensor):
    taco_tensor = '-' + einsum_tensor.name if einsum_tensor.negated else einsum_tensor.name
    taco_tensor = taco_tensor + '(' + einsum_tensor.indexing + ')' if einsum_tensor.order > 0 else taco_tensor
    return taco_tensor

  taco_prog = gen_taco_tensor(einsum_prog.lhs_tensor) + ' = '
  if einsum_prog.n_tensors == 2:
    return taco_prog + gen_taco_tensor(einsum_prog.rhs_tensors[0])
  else:
    for i in range(einsum_prog.n_arithops):
      taco_prog += gen_taco_tensor(einsum_prog.rhs_tensors[i]) + ' ' + str(einsum_prog.arithops[i]) + ' '
    taco_prog += gen_taco_tensor(einsum_prog.rhs_tensors[einsum_prog.n_tensors - 2])
  return taco_prog

