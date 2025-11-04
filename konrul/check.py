import glob
import json
import os
import subprocess

import konrul.constants as constants
import konrul.einsum_program as einsum_program
import konrul.literal_constant_scan as literal_constant_scan
import konrul.lowering as lowering


class Checker():
  
  def __init__(self, program_path, original_lists = None):
    self.lower = lowering.Lower(program_path)
    self.original_lists = original_lists

  
  def __wagner_fischer(seq1, seq2):
    # Compute Levensthein edit distance using Wagner-Fischer algorithm
    # Create (n + 1) x (m + 1) matrix to store sub-problem solutions 
    # We need an extra row and column to store base cases
    edit_distance_matrix = [[float('inf')] * (len(seq2) + 1) for _ in range(len(seq1) + 1)]
    
    for i in range(len(seq2) + 1):
      edit_distance_matrix[len(seq1)][i] = len(seq2) - i
  
    for i in range(len(seq1) + 1):
      edit_distance_matrix[i][len(seq2)] = len(seq1) - i
  
    for i in range(len(seq1) - 1, -1, -1):
      for j in range(len(seq2) - 1, -1, -1):
        if seq1[i] == seq2[j]:
          edit_distance_matrix[i][j] = edit_distance_matrix[i+1][j+1]
        else:
          edit_distance_matrix[i][j] = 1 + min(edit_distance_matrix[i+1][j], edit_distance_matrix[i][j+1], edit_distance_matrix[i+1][j+1])
       
    return edit_distance_matrix[0][0]
  
  
  def __program_similarity(prog1, prog2):
    if not prog1 and not prog2:
      return 1.0
    
    if not prog1:
      return 0.0
    
    if not prog2:
      return 0.0 
   
    prog1_str = ''.join(prog1)
    prog2_str = ''.join(prog2)
    return max(1 - (Checker.__wagner_fischer(prog1_str, prog2_str) / len(prog2_str)), 0.0)
  
  
  def __complete_indexing_list(indexing_list, variable_list):
    # Ignore lhs for the indexing analysis always returns an indexing for that
    for i in range(1, len(variable_list)):
      # If a variable is a scalar, it does not have a corresponding indexing
      if not variable_list[i].startswith('t_'):
        indexing_list.insert(i, '')
        
  
  def __normalize_einsum_indexing(indexing):
    base = ord('i')
    indexing_tokens = indexing.split()
    einsum_indexing = []
  
    # Polly returns expressions with constants in case of linearized array
    # accesses with constant loop bounds. We order the variables in the 
    # indexing expression by the constant offsets
    max_offset = 0
    for tok in indexing_tokens:
      if tok == '+' or tok == '-' or tok == '*':
        continue
  
      if tok.endswith(','):
        tok = tok.rstrip(',')
      
      offset, var_idx = tok.split('i')
      if offset.isnumeric():
        if int(offset) > max_offset:
          einsum_indexing.insert(0, chr(base + int(var_idx)))
          max_offset = int(offset)
        else:
          einsum_indexing.append(chr(base + int(var_idx)))
      else:
        einsum_indexing.append(chr(base + int(var_idx)))
  
    return ','.join(einsum_indexing)
  
  
  def __parse_scop_file(statements, arrays):
    indexing_rhs = []
    indexing_lhs = ''
    for statement in statements:
      accesses = statement['accesses']
      for access in accesses:
        relation = access['relation']
        array = relation.split('->')[-1][1:-2]
        name = array[:array.find('[')]
        indexing = array[array.find('[') + 1 : array.find(']')]
        if name in arrays:
          if access['kind'] == 'write':
            # TACO does not allow the same variable on both sides of assignment. If that is the case, normalize the reference 
            # on the left-hand side
            indexing_lhs = Checker.__normalize_einsum_indexing(indexing)
          else:          
            indexing_rhs.append(Checker.__normalize_einsum_indexing(indexing))
  
    return [indexing_lhs] + indexing_rhs
  
  
  def __build_indexing_list(polly_kernel_name):
    jscop = glob.glob(polly_kernel_name + '*__*.jscop')
    if not jscop:
      print('Polly failed to detect scops for ', polly_kernel_name)
      return ['']
    
    # Polly deceted more than one scop file
    jscop_file_name = jscop[0] if len(jscop) == 1 else jscop[-1]
      
    with open(jscop_file_name, 'r') as jscop_file:
      polly_scops = json.load(jscop_file)
      arrays = [array['name'] for array in polly_scops['arrays']]
      return Checker.__parse_scop_file(polly_scops['statements'], arrays)
  
  
  def __get_indexing_position_list(indexing_vars, indexing_list):
    indexing_pos_list = []
    for var in indexing_vars:
      for indexing in indexing_list:
        indexing_pos_list.append('0' if var not in indexing else str(indexing.index(var) + 1))
        
    return indexing_pos_list
  
  
  def __compute_indexing_score(self, candidate_variable_list):
    self.lower.run_polly()
    candidate_indexing_list = Checker.__build_indexing_list('compute')
    for f in glob.glob('*.jscop'):
      os.remove(f)
      
    if len(candidate_indexing_list) != len(candidate_variable_list):
      Checker.__complete_indexing_list(candidate_indexing_list, candidate_variable_list)
    
    # To get unique indexing variables, we use dict.fromkeys insted of set to preserve the order
    orig_indexing_vars = list(dict.fromkeys(''.join(self.original_lists['indexing']).replace(',', '')))
    candidate_indexing_vars = list(dict.fromkeys(''.join(candidate_indexing_list).replace(',', '')))
    
    orig_indexing_pos_list = Checker.__get_indexing_position_list(orig_indexing_vars, self.original_lists['indexing'])
    candidate_indexing_pos_list = Checker.__get_indexing_position_list(candidate_indexing_vars, candidate_indexing_list)
    
    return Checker.__program_similarity(candidate_indexing_pos_list, orig_indexing_pos_list)
  
  
  def compute_score(self, candidate):
    try:
      self.lower.compile_TACO(candidate)
    except subprocess.CalledProcessError:
      candidate.score = 0.0
      return
  
    # Variable and indexing similarity
    try:
      candidate_variable_list = self.lower.run_LLVM_pass(constants.VAR_SIM_PASS, 'var-sim')
      # If there are constants, insert them at the beginning of the naming trace 
      for const in literal_constant_scan.get_relevant_constants(self.lower.c_program_path):
        candidate_variable_list.insert(1, const)
  
      # Indexing similarity
      # We compute it first as we need information from the similarity pass regarding
      # each tensor being a scalar or not
      candidate.indexing_sim = self.__compute_indexing_score(candidate_variable_list)
      
      candidate_variable_list = [var if var.isnumeric() else var[-1] for var in candidate_variable_list]
      candidate.variable_sim = Checker.__program_similarity(candidate_variable_list, self.original_lists['variable'])
    except subprocess.CalledProcessError:
      # Return very low value for score in case analysis crashes
      candidate.score = 0.0
      return
      
    # Arithmetic Operators similarity
    try:
      candidate_arithop_list = self.lower.run_LLVM_pass(constants.ARITHOP_SIM_PASS, 'arith-op-sim')
      candidate.arithop_sim = Checker.__program_similarity(candidate_arithop_list, self.original_lists['arithop'])
    except subprocess.CalledProcessError:
      # Return very low value for score in case analysis crashes
      candidate.score = 0.0
      return
    
    candidate.score = candidate.arithop_sim * candidate.variable_sim * candidate.indexing_sim

  
  def check(self, candidate: einsum_program.EinsumProgram) -> bool:
    return True if candidate.score == 1.00 else False
  

  @staticmethod
  def get_original_lists(program_path):
    checker = Checker(program_path)
    checker.lower.lower_to_LLVM()
    checker.lower.run_polly()
    
    lists = dict()
    lists['variable'] = checker.lower.run_LLVM_pass(constants.VAR_SIM_PASS, 'var-sim')
    # If there are constants, insert them at the beginning of the naming trace 
    for const in literal_constant_scan.get_relevant_constants(program_path):
      lists['variable'].insert(1, const)
      
    lists['arithop'] = checker.lower.run_LLVM_pass(constants.ARITHOP_SIM_PASS, 'arith-op-sim')
    lists['indexing'] = Checker.__build_indexing_list(os.path.splitext(os.path.basename(program_path))[0])
      
    # Fullfill scalar indexing
    if len(lists['indexing']) != len(lists['variable']):
      Checker.__complete_indexing_list(lists['indexing'], lists['variable'])
        
    # Normalize names
    lists['variable'] = [var if var.isnumeric() else var[-1] for var in lists['variable']]
  
    return lists
  