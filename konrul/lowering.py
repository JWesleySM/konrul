import re
import os
import subprocess

from konrul.constants import TACO, CLANG, LLVM, POLLY_LLVM
import konrul.export as export

def is_valid(taco_prog):
  if len(taco_prog.split(' = ')) != 2:
    return False
  
  if taco_prog.count('=') > 1:
    return False
  
  n_rhs_tensors = 0
  n_binops = 0 
  rhs = taco_prog.split(' = ')[1]
  for elem in rhs.split():
    if re.match('^-?[b-z]|-?Cons*', elem):
      n_rhs_tensors += 1
    elif elem in ['+', '-', '*', '/']:
      n_binops += 1
    else:
      return False

  
  if n_binops != n_rhs_tensors - 1:
    return False

  def has_constant_token():
    for elem in taco_prog.split():
      if elem.startswith('Cons'):
        return True

    return False


  def insert_consts():
    taco_prog_with_cons = taco_prog
    for elem in taco_prog.split():
      if elem.startswith('Cons'):
        taco_prog_with_cons = taco_prog_with_cons.replace(elem, str(1))
  
    return taco_prog_with_cons
    
    
  if(has_constant_token()):
    taco_prog = insert_consts()

  try:
    arguments = [TACO, taco_prog, '--print-nocolor', '--write-source=taco_temp.c']
    compile_command = subprocess.run(arguments, check = True, stderr = open('/dev/null' ,'w'), stdout = open('/dev/null' ,'w'))
    compile_command = subprocess.run([CLANG, '-c', 'taco_temp.c'], check = True, stderr = open('/dev/null', 'w'))
  except subprocess.CalledProcessError as e:
    return False

  return True

TACO_PREAMBLE = '#include <stdint.h>\n \
#ifndef TACO_TENSOR_T_DEFINED\n\
#define TACO_TENSOR_T_DEFINED\n\
typedef enum { taco_mode_dense, taco_mode_sparse } taco_mode_t;\n\
typedef struct {\n\
  int32_t      order;         // tensor order (number of modes)\n\
  int32_t*     dimensions;    // tensor dimensions\n\
  int32_t      csize;         // component size\n\
  int32_t*     mode_ordering; // mode storage ordering\n\
  taco_mode_t* mode_types;    // mode storage types\n\
  uint8_t***   indices;       // tensor index data (per mode)\n\
  uint8_t*     vals;          // tensor values\n\
  uint8_t*     fill_value;    // tens/hp  or fill value\n\
  int32_t      vals_size;     // values array size\n\
} taco_tensor_t;\n\
#endif\n'

class Lower():
  
  def __init__(self, program_path):
    working_dir = os.path.dirname(program_path)
    c_program = os.path.basename(program_path)
    lowered = os.path.splitext(c_program)[0] + '.bc'
    polly_lowered = os.path.splitext(lowered)[0] + '.polly.bc'
    self.c_program_path = program_path
    self.lowered_path = os.path.join(working_dir, lowered)
    self.polly_lowered_path = os.path.join(working_dir, polly_lowered)
    # TODO: 
    #self.clang
    #self.llvm
    #self.polly_llvm
    #self.taco
    #self.passes


  def run_polly(self):
    # canonicalize
    try:
      arguments = [POLLY_LLVM + '/opt', '-polly-canonicalize', self.lowered_path, '-o', self.polly_lowered_path]
      subprocess.run(arguments, check = True, stderr = subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
      print('Error canonicalizing with Polly', e)
      raise e
  
    # run polly and generate scops
    try:
      arguments = [POLLY_LLVM + '/opt', '-basic-aa', '-polly-use-llvm-names', '-polly-export-jscop', self.polly_lowered_path,
                   '-polly-process-unprofitable']
      subprocess.run(arguments, check = True, stderr = subprocess.STDOUT, stdout = open('/dev/null', 'w'))
    except subprocess.CalledProcessError as e:
      print('Error running Polly', e)
      raise e
    

  def lower_to_LLVM(self):
    try:
      arguments = [CLANG, '-c', '-Oz', '-fno-discard-value-names', '-emit-llvm', self.c_program_path, '-o', self.lowered_path]
      subprocess.run(arguments, check = True, stderr = subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
      print('Error lowering to LLVM', e)
      raise e
    

  def run_LLVM_pass(self, LLVM_pass, LLVM_pass_cli):
    try:
      arguments = [f'{LLVM}opt', '--load-pass-plugin', LLVM_pass, f'-passes={LLVM_pass_cli}', '-disable-output', self.lowered_path]
      command = subprocess.run(arguments, check = True, capture_output = True)
      return command.stdout.decode('utf-8').split()
  
    except subprocess.CalledProcessError as e: 
      print(f'Error running {LLVM_pass}', e)
      raise e
    

  def compile_TACO(self,einsum_prog):
    # Compile to C using TACO compiler
    with open(self.c_program_path, 'w') as taco_file:
      taco_file.write(TACO_PREAMBLE)
      taco_file.flush()
     
      arguments = [TACO, export.taco_gen(einsum_prog), '--print-nocolor']
      for tensor in [einsum_prog.lhs_tensor] + einsum_prog.rhs_tensors:
        arguments += [f'-t={tensor.name}:int']

      try:
        subprocess.run(arguments, check = True, stdout = taco_file, stderr = subprocess.STDOUT)
      except subprocess.CalledProcessError as e:
        print('Error compiling TACO program', e)
        raise e


    # Compile to LLVM
    try:
      self.lower_to_LLVM()
    except subprocess.CalledProcessError as e:
      raise e 
    