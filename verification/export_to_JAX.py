import re

def _is_bloating(einsum_prog):
  if einsum_prog.n_tensors == 2:
    return True if einsum_prog.lhs_tensor.order > einsum_prog.rhs_tensors[0].order else False
  else:
    return False


def _is_initialization(einsum_prog):
  if einsum_prog.n_tensors == 2:
    return True if einsum_prog.rhs_tensors[0].order == 0 else False
  else:
    return False


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
  if einsum_prog.n_tensors == 2:
    return True if einsum_prog.lhs_tensor.order <= einsum_prog.rhs_tensors[0].order else False
  elif einsum_prog.n_tensors == 3:
    if einsum_prog.rhs_tensors[0].name.isnumeric():
      return False
    
    if einsum_prog.lhs_tensor.order > einsum_prog.rhs_tensors[0].order:
      return False
    
    if einsum_prog.arithops[0] != '*':
      return False
    
    return True
  else:
    if all(arithop == '*' for arithop in einsum_prog.arithops):
      return True

  return False


def _jnp_init_codegen(einsum_prog):
  init_value = einsum_prog.rhs_tensors[0]
  if init_value.name.startswith("Cons"):
    return f"jnp.einsum('i->i', {einsum_prog.lhs_tensor.name}) * {init_value.name}"
    #return f"jnp.full({einsum_prog.lhs_tensor.name}.shape, {init_value}, dtype = float)"
  else:
    return "jnp.full_like(" + einsum_prog.lhs_tensor.name + "," + init_value.name + ")"


def _jnp_bloat_codegen(einsum_prog):
  rhs_tensor = einsum_prog.rhs_tensors[0]
  return f"jnp.broadcast_to({rhs_tensor.name}, {einsum_prog.lhs_tensor.name}.shape).transpose(1,0)"


def _einsum_indexing_codegen(indexing_expr):
  return indexing_expr.replace(',','') if indexing_expr else ''
  

def _jnp_einsum_codegen(lhs_tensor, einsum_tensors):
  einsum_exp = '\'' + _einsum_indexing_codegen(einsum_tensors[0].indexing)
  einsum_args = '-' + einsum_tensors[0].name if einsum_tensors[0].negated else einsum_tensors[0].name
  for t in einsum_tensors[1:]:
    einsum_exp += ',' + _einsum_indexing_codegen(t.indexing)
    t_name = '-' + t.name if t.negated else t.name
    einsum_args += ',' + t_name
  
  einsum_exp += '->' + _einsum_indexing_codegen(lhs_tensor.indexing) + '\''
  return einsum_exp, einsum_args


def _jnp_pure_einsum_codegen(einsum_prog):
  einsum_call = 'jnp.einsum('
  einsum_exp, einsum_args = _jnp_einsum_codegen(einsum_prog.lhs_tensor, einsum_prog.rhs_tensors)
  return einsum_call + einsum_exp + ',' + einsum_args + ')'


def _jnp_pluseq_einsum_codegen(lhs_tensor, einsum_tensors, pluseq_tensor):
  einsum_call = 'jnp.einsum('
  einsum_exp, einsum_args = _jnp_einsum_codegen(lhs_tensor, einsum_tensors)
  return pluseq_tensor.name + " + " + einsum_call + einsum_exp + ',' + einsum_args + ', optimize="greedy")'


def _jnp_extended_einsum_codegen(einsum_prog):
  if einsum_prog.n_tensors == 1:
    tensor = einsum_prog.rhs_tensors[0]
    tensor_name = '-' + tensor.name if tensor.negated else tensor.name
    jnp_prog = einsum_prog.lhs_tensor.name + ' = ' + tensor_name
  else:
    jnp_prog = ""
    for i in range(einsum_prog.n_arithops):
      if i in einsum_prog.l_par:
        jnp_prog += einsum_prog.l_par[i]
      tensor = einsum_prog.rhs_tensors[i]
      tensor_name = '-' + tensor.name if tensor.negated else tensor.name
      jnp_prog += tensor_name
      if i in einsum_prog.r_par:
        jnp_prog += einsum_prog.r_par[i]
      jnp_prog +=  ' ' + einsum_prog.arithops[i] + ' '

    tensor = einsum_prog.rhs_tensors[-1]
    tensor_name = '-' + tensor.name if tensor.negated else tensor.name
    jnp_prog += tensor_name
    if einsum_prog.n_tensors - 2 in einsum_prog.r_par:
      jnp_prog += einsum_prog.r_par[einsum_prog.n_tensors - 2]
  
  return jnp_prog


def jnp_gen(einsum_prog):
  if _is_initialization(einsum_prog):
    jnp_prog = _jnp_init_codegen(einsum_prog)
  elif _is_bloating(einsum_prog):
    jnp_prog = _jnp_bloat_codegen(einsum_prog)
  elif _is_pure_einsum(einsum_prog):
    jnp_prog = _jnp_pure_einsum_codegen(einsum_prog)
  else:
    pluseq_einsum, pluseq_tensor_idx = _is_pluseq_einsum(einsum_prog)
    if pluseq_einsum:
      pluseq_tensor = einsum_prog.rhs_tensors[pluseq_tensor_idx]
      einsum_tensors = einsum_prog.rhs_tensors[1:] if pluseq_tensor_idx == 0 else einsum_prog.rhs_tensors[:-1]
      jnp_prog = _jnp_pluseq_einsum_codegen(einsum_prog.lhs_tensor, einsum_tensors, pluseq_tensor)
    else:
      jnp_prog = _jnp_extended_einsum_codegen(einsum_prog)
  return jnp_prog


def export_einsum_to_JAX(einsum_program, args_order, substitution):
  kernel_exp = jnp_gen(einsum_program)    
  lambda_vars = ",".join(args_order)
  
  # To ensure that we replace each template variable in the TACO program, we create 
  # placeholders for each variable in the substitution and perform the substitution 
  # according to it to the placeholders order
  placeholders = {template_var: f"__PLACEHOLDER_{i}__" for i, (_, template_var) in enumerate(substitution)}
  for template_var, placeholder in placeholders.items():
    kernel_exp = re.sub(rf"\b{re.escape(template_var)}\b", placeholder, kernel_exp)
    
  for (input, template_var), placeholder in zip(substitution, placeholders.values()):
    kernel_exp = kernel_exp.replace(placeholder, input)
    
  # If this is a benchmark that just inits a tensor, we need to also replace
  # the output variable in the lambda function body
  if _is_initialization(einsum_program):
    pattern = rf"\b{re.escape(einsum_program.lhs_tensor.name)}\b"
    input_vars = {subs[0] for subs in substitution} 
    for arg in args_order:
      if arg not in input_vars:
        kernel_exp = re.sub(pattern, arg, kernel_exp)

  print(f"lambda {lambda_vars}: {kernel_exp}")
  return lambda_vars, kernel_exp
