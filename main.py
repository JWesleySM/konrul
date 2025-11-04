import argparse
import os
import time

import konrul.export as export
import konrul.guess as guess
import konrul.io_handler as io_handler
import konrul.io_test as io_test
import konrul.search as search

from konrul.check import Checker
import verification.verify as verifier


def lift(original_program, io, original_mlir, n_iter):
  original_lists = Checker.get_original_lists(original_program)

  pick_initial_start_time = time.time()
  initial_candidate = guess.guess(original_program)
  pick_initial_time = time.time() - pick_initial_start_time

  print(original_program)
  print(original_lists)
  print(initial_candidate)
  print(20 * '-')

  checker = Checker(os.path.join(os.path.dirname(original_program), 'candidate.c'), original_lists)
  checker.compute_score(initial_candidate)
  search._print_similarities(initial_candidate)

  io_set = io_handler.parse_io(io)
  if checker.check(initial_candidate): 
    test_code, substitution = io_test.io_test(initial_candidate, io_set)
    if test_code == io_test.TestingReturnCode.SUCCESS:
      verified = verifier.verify(original_mlir, candidate, substitution)        
      return initial_candidate, 0, verified, pick_initial_time, 0.0, 0.0

  total_explored = 0
  search_time = 0.0
  test_time = 0.0
  for _ in range(n_iter):
    search_time_start = time.time()
    candidates, n_explored = search.search(initial_candidate, original_lists, checker)
    search_time += time.time() - search_time_start
    #candidates , n_explored = ([EinsumProgram.from_string('a(i,j) = b(i) + c * d(i,j)')], 0)
    test_time_start = time.time()
    for candidate in candidates:
      print('Returned by search:', candidate)
      test_return_code, substitution = io_test.io_test(candidate, io_set)
      test_time += time.time() - test_time_start
      print(test_return_code)
      if test_return_code == io_test.TestingReturnCode.SUCCESS:
        verified = verifier.verify(original_mlir, candidate, substitution)
        return candidate, n_explored, verified, pick_initial_time, search_time, test_time
      
    total_explored += n_explored
      
  return None, total_explored, pick_initial_time, search_time, test_time, False
  

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('original', type = str, help = 'Path to the program to be lifted')
  parser.add_argument('io', type = str, help = 'Path to the input-output file')
  parser.add_argument('mlir', type = str, help = 'Path to the MLIR version of the program to be lifted')
  parser.add_argument('n_iter', type = int, help = 'Maximum number of iterations')
  parser.parse_args()
  args = parser.parse_args()
  
  assert os.path.isfile(args.original), f'Could not find original program {args.original}'
  assert os.path.isfile(args.io), f'Could not find IO file {args.io}'
  assert os.path.isfile(args.mlir), f'Could not find MLIR file {args.mlir}'
  
  lifted , n_explored, verified, start_time, search_time, test_time = lift(args.original, args.io, args.mlir, args.n_iter)
  print("\n" + 20 * '-')
  print('Answer:', lifted)
  print('# explored: ', n_explored)
  print('Pick initial time: ', start_time)
  print('Searching time:', search_time)
  print('Checking time:', test_time)
  if lifted:
    print("Verification:", verified)
    print('PyTorch:', export.pytorch_gen(lifted))
  

if __name__ == '__main__':
  main()