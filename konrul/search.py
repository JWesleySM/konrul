import enum
import random
import string
import time

import konrul.einsum_program as einsum_program

class EditRule(enum.Enum):
  EDIT_ARITHOP = 0
  EDIT_INDEXING = 1
  REPLACE_ID = 2
  ADD_TENSOR = 3
  DEL_TENSOR = 4
  SWAP_SIGN = 5

ARITH_OPS = ['+', '-', '*', '/']

# Index variables lie in the range i-j-k-l-m-n
IDX_VARIABLES_ASCII = [8,9,10,11,12,13]

LLVM_to_Einsum = {
  'add'  : '+',
  'sub'  : '-',
  'mul'  : '*',
  'sdiv' : '/'
}


def _is_legal_program(candidate, new_arithop, lhs_arithop, rhs_arithop):
  if new_arithop == '*':
    return True
  
  if candidate.lhs_tensor.order == 0:
    return not any(tensor.order > 0 for tensor in [candidate.lhs_tensor] + candidate.rhs_tensors)
  
  ''' In face of division the indexing of both tensors must be the 
  same for non-scalar tensors
  '''
  if new_arithop == '/':
    if lhs_arithop.name.isnumeric() or rhs_arithop.name.isnumeric():
      return True
    
    if rhs_arithop.indexing == '':
      return True
    
    return lhs_arithop.indexing == rhs_arithop.indexing
    
  return True


# TODO: Adjust this method
def _should_swap_sign(candidate, orig_arithop_list):
  if 'sub' in orig_arithop_list:
    return True
  elif candidate.has_tensor_negated():
    return True
      
  return False


def _change_variable(edit_rule):
  return edit_rule == EditRule.ADD_TENSOR or edit_rule == EditRule.DEL_TENSOR or edit_rule == EditRule.REPLACE_ID


def _duplicate_tensor_reference(candidate, new_arithop):
  new_arithop = LLVM_to_Einsum[new_arithop]
  tensor_ref_idx = random.randint(0, candidate.n_tensors - 2)
  tensor_ref = candidate.rhs_tensors[tensor_ref_idx]
  new_tensor = tensor_ref.copy()
  return einsum_program.EinsumProgram(candidate.lhs_tensor.copy(), candidate.rhs_tensors.copy() + [new_tensor], candidate.arithops.copy() + [new_arithop], edit_rule = EditRule.ADD_TENSOR)


def _add_tensor_reference(candidate, original_indexing, original_arithop):
  # Get a new tensor name excluding the index variables 
  new_name = string.ascii_lowercase[random.choice([i for i in range(1, 25) if i not in IDX_VARIABLES_ASCII])]
  if new_name in candidate.tensor_names:
    new_tensor = candidate.get_reference_to_tensor(new_name)
  else:
    # In case we cannot get the indexing that is likely to be the missing reference,
    # select a random one from the original list
    if candidate.n_tensors > len(original_indexing):
      new_indexing = random.choice(original_indexing)
    else:
      new_indexing = original_indexing[candidate.n_tensors]
    new_order = 0 if not new_indexing else new_indexing.count(',') + 1
    new_tensor = einsum_program.Tensor(new_name, new_order, new_indexing, False)
  
  # get only the arithmetic operators from list (exclude operands)
  original_arithop_list = [arithop for arithop in original_arithop if arithop in LLVM_to_Einsum]
  if original_arithop:
    new_op = LLVM_to_Einsum[random.choice(original_arithop_list)]
  else:
    new_op = random.choice(ARITH_OPS)
 
  return einsum_program.EinsumProgram(candidate.lhs_tensor.copy(), candidate.rhs_tensors.copy() + [new_tensor], candidate.arithops.copy() + [new_op], edit_rule = EditRule.ADD_TENSOR)
   
       
def _edit_tensor_reference(candidate, orig_naming):
  tensor_ref_idx = random.randint(0, candidate.n_tensors - 2)
  tensor_ref = candidate.rhs_tensors[tensor_ref_idx] 
  new_name = orig_naming[tensor_ref_idx]
  while new_name == tensor_ref.name:
    tensor_ref_idx = random.randint(0, candidate.n_tensors - 2)
    new_name = orig_naming[tensor_ref_idx]
    
  # If the name is already present in the candidate, just duplicate it. 
  # Otherwise create a new tensor
  if new_name in candidate.tensor_names:
    new_tensor = candidate.get_reference_to_tensor(new_name).copy()
  else:
    tensor_ref = candidate.rhs_tensors[tensor_ref_idx]
    new_order, new_indexing = (0, '') if new_name.isnumeric() else (tensor_ref.order, tensor_ref.indexing)
    new_tensor = einsum_program.Tensor(new_name, new_order, new_indexing, tensor_ref.negated)
  
  new_rhs_tensors = candidate.rhs_tensors.copy()
  new_rhs_tensors[tensor_ref_idx] = new_tensor
  return einsum_program.EinsumProgram(candidate.lhs_tensor.copy(), new_rhs_tensors, candidate.arithops.copy(), edit_rule = EditRule.REPLACE_ID)
  

def _delete_tensor_reference(candidate):
  assert candidate.n_tensors > 2, 'Cannot delete tensor reference in programs with only one tensor on the right-hand side'

  tensor_ref_idx = random.randint(1, candidate.n_tensors - 2)
  if tensor_ref_idx == candidate.n_tensors - 2:
    arithop_idx = tensor_ref_idx - 1
  else:
    arithop_idx = random.choice([tensor_ref_idx, tensor_ref_idx - 1])
 
  new_rhs_tensors = [candidate.rhs_tensors[i].copy() for i in range(candidate.n_tensors - 1) if i != tensor_ref_idx]
  new_arithops = [candidate.arithops[i] for i in range(candidate.n_arithops) if i != arithop_idx]
  
  return einsum_program.EinsumProgram(candidate.lhs_tensor.copy(), new_rhs_tensors, new_arithops, edit_rule = EditRule.DEL_TENSOR)


def _edit_indexing(candidate, orig_indexing_list):
  if candidate.lhs_tensor.indexing != orig_indexing_list[0]:
    tensor_ref_idx = 0
    tensor_ref = candidate.lhs_tensor
  else:
    try:
      tensor_ref_idx = random.randint(1, candidate.n_tensors - 1)
      tensor_ref = candidate.rhs_tensors[tensor_ref_idx - 1]

      while tensor_ref.indexing == orig_indexing_list[tensor_ref_idx]:
        tensor_ref_idx = random.randint(1, candidate.n_tensors - 1)
        tensor_ref = candidate.rhs_tensors[tensor_ref_idx - 1]

    except IndexError:
      # If there is no indexing for the selected tensor, it might be a case of
      # repeated references in the guess but not in the candidate
      return _delete_tensor_reference(candidate)
       
  new_indexing = orig_indexing_list[tensor_ref_idx]
  new_order = 0 if not new_indexing else new_indexing.count(',') + 1
  new_tensor = einsum_program.Tensor(tensor_ref.name, new_order, new_indexing, tensor_ref.negated)
  
  if tensor_ref_idx == 0:
    new_candidate = einsum_program.EinsumProgram(new_tensor, candidate.rhs_tensors.copy(), candidate.arithops.copy(), edit_rule = EditRule.EDIT_INDEXING)
  else:
    new_rhs_tensors = candidate.rhs_tensors.copy()
    new_rhs_tensors[tensor_ref_idx - 1] = new_tensor
    new_candidate = einsum_program.EinsumProgram(candidate.lhs_tensor.copy(), new_rhs_tensors, candidate.arithops.copy(), edit_rule = EditRule.EDIT_INDEXING) 

  return new_candidate 


def _edit_arithop(candidate, orig_arithop):
  assert candidate.n_tensors > 2, 'Cannot edit binary operator: candidate with only one tensor on the right-hand side'
 
  arithop_idx = random.randint(0, candidate.n_arithops - 1)
  new_arithop = LLVM_to_Einsum[random.choice(orig_arithop)] if orig_arithop else random.choice(ARITH_OPS)

  aithop_timeout = time.time() + 3
  while candidate.arithops[arithop_idx] == new_arithop:
    if time.time() > aithop_timeout:
      return None
      
    # If there is only once choice of binary operator and it is equal to the selected one
    # the edit_rule cannot be applied.
    
    # We can try to things:
    # Swap a sign if there is a subtraction in the original code or if there is a 
    # negate tensor in the candidate but not on the original list
    if _should_swap_sign(candidate, orig_arithop):
      return _swap_sign(candidate)
  
    new_arithop = LLVM_to_Einsum[random.choice(orig_arithop)] if orig_arithop else random.choice(ARITH_OPS)
  
  new_arithops = candidate.arithops.copy()
  new_arithops[arithop_idx] = new_arithop
  new_candidate = einsum_program.EinsumProgram(candidate.lhs_tensor.copy(), candidate.rhs_tensors.copy(), new_arithops, edit_rule = EditRule.EDIT_ARITHOP)
  return new_candidate if _is_legal_program(new_candidate, new_arithop, new_candidate.rhs_tensors[arithop_idx], new_candidate.rhs_tensors[arithop_idx + 1]) else None
  

def _swap_sign(candidate):
  # We edit the first reference only if it is a constant
  tensor_ref_idx = random.randint(0, candidate.n_tensors - 2)
  tensor_ref = candidate.rhs_tensors[tensor_ref_idx]
  new_tensor = einsum_program.Tensor(tensor_ref.name, tensor_ref.order, tensor_ref.indexing, not tensor_ref.negated)
  new_rhs_tensors = candidate.rhs_tensors.copy()
  new_rhs_tensors[tensor_ref_idx] = new_tensor
  return einsum_program.EinsumProgram(candidate.lhs_tensor.copy(), new_rhs_tensors, candidate.arithops.copy(), edit_rule = EditRule.SWAP_SIGN) 


def _edit_variable(candidate, original_lists):
  if candidate.n_tensors < len(original_lists['variable']):
    return _add_tensor_reference(candidate, original_lists['indexing'], original_lists['arithop'])
  elif candidate.n_tensors > len( original_lists['variable']):
    return _delete_tensor_reference(candidate)
  else:    
    return _edit_tensor_reference(candidate, original_lists['variable'][1:])


def _apply_edit_rule(candidate, original_lists, checker):
  if candidate.variable_sim != 1.0 and candidate.tensor_names != original_lists['variable']:
    if candidate.edit_rule:
      # We only attempt to edit the naming dimension of a program in case
      # its score was harmed by a edit rule that changes variables.
      # This is safe since the naming edit rules have priority
      # If an edit rule other than naming caused a decrease in the naming score,
      # that is likely to mean a case where the names differ at the LLVM level
      # but are correct in the Einsum space
      if _change_variable(candidate.edit_rule):
        return _edit_variable(candidate, original_lists)
    else:
      return _edit_variable(candidate, original_lists)
        
  if candidate.indexing_sim != 1.0:
    # We need to check if the indexings are the same even if the indexing score
    # is 1.0. That happens because there might be duplicated variables or
    # variables inserted by the compiler when we look at the LLVM space
    if candidate.indexings != original_lists['indexing']:
      return _edit_indexing(candidate, original_lists['indexing'])     
  
  orig_arithops = [arithop for arithop in original_lists['arithop'] if arithop in LLVM_to_Einsum]
  edited = None
  if candidate.arithop_sim != 1.0:
    if candidate.n_tensors > 2:
      if candidate.n_arithops < len(orig_arithops):
        edited1 = _edit_arithop(candidate, orig_arithops)
        edited2 = _duplicate_tensor_reference(candidate, orig_arithops[candidate.n_arithops])
        if edited1 and edited2:
          checker.compute_score(edited1)
          checker.compute_score(edited2)
          edited = edited1 if edited1.score >= edited2.score else edited2
        else:
          edited = edited1 if edited2 is None else edited2
        
      else:
        edited = _edit_arithop(candidate, orig_arithops)
             
    elif _should_swap_sign(candidate, orig_arithops):
      edited = _swap_sign(candidate)
    
    if edited is None:
      if candidate.n_arithops < len(orig_arithops):
        return _duplicate_tensor_reference(candidate, orig_arithops[candidate.n_arithops])
      else:
        return _delete_tensor_reference(candidate)
    else:
      return edited
   
  # In case of duplicated references, the variable similarity may be perfect but the einsum
  # program still does not have the correct number of references
  if candidate.variable_sim == 1.0 and candidate.tensor_names != original_lists['variable']:
    return _edit_variable(candidate, original_lists)
    
  # Diferent operations might become reorganized in LLVM in a way that are equal at the LLVM
  # level but are different at Einsum level
  if candidate.arithop_sim == 1.0 and candidate.n_tensors > 2:
    return _edit_arithop(candidate, orig_arithops)
         
  return candidate


def search(initial_candidate, original_lists, checker):
  timeout = time.time() + 30
  visited = set(str(initial_candidate).replace(' ', ''))
  similar_candidates = []

  candidate = initial_candidate
  while time.time() < timeout:
    next_candidate = _apply_edit_rule(candidate, original_lists, checker)

    if not next_candidate:
      continue
    if str(next_candidate) in visited:
      continue   
    
    visited.add(str(next_candidate))
    checker.compute_score(next_candidate)
    
    if str(next_candidate) == str(candidate):
      return [candidate], len(visited)
 
    print('Candidate: ', next_candidate, "Last edit:", next_candidate.edit_rule, next_candidate.score, "\n")
    
    if checker.check(next_candidate):
      return [next_candidate], len(visited)
    else:
      if next_candidate.variable_sim >= 0.66 and next_candidate.arithop_sim >= 0.5 and next_candidate.indexing_sim >= 0.66:
        similar_candidates.append(next_candidate)
      elif next_candidate.variable_sim == 1.0 and next_candidate.indexing_sim == 1.0 and candidate.arithop_sim != 1.0:
        similar_candidates.append(next_candidate)

    candidate = next_candidate    
  if similar_candidates:
    return similar_candidates, len(visited)
  
  return [candidate], len(visited)
  
 
