import clang.cindex as clang

LOOP_TYPES = [clang.CursorKind.FOR_STMT, clang.CursorKind.WHILE_STMT, clang.CursorKind.DO_STMT] 

def _parse_program(program_path):
  """Parse a C program and return the corresponding translation unit."""
  idx = clang.Index.create()
  tu = idx.parse(program_path, args = ['-c'])
  return tu


def _extract_clang(cursor):
  """ Extract the text from the source code that corresponds to the AST
  cursor.
  """
  if cursor is None:
    return ''
  filename = cursor.location.file.name
  with open(filename, 'r') as fh:
    contents = fh.read()
  return contents[cursor.extent.start.offset: cursor.extent.end.offset]


def _get_variable_name(cursor):
  for c in cursor.walk_preorder():
    if c.kind == clang.CursorKind.DECL_REF_EXPR:
      return c.get_definition().spelling
      
  return ''


def _get_nodes_by_kind(tu, kinds):
  """ Return a list of nodes with the given kinds."""
  return [n for n in tu.cursor.walk_preorder() if n.kind in kinds]


def _get_loop_control_vars(tu):
  """ Get the variables that are used as control variables in all the loops
  found in source code.
  """
  loops = _get_nodes_by_kind(tu, LOOP_TYPES)
  loop_control_vars = set()
  for l in loops:
    # We assume that the loops is the classical form, therefore, the loop condition
    # is the second element in the list formed by the children of the loop node in the AST.
    # Variables are saved using their hash to avoid duplicates.
    loop_cond = list(l.get_children())[1]
    if loop_cond.kind == clang.CursorKind.BINARY_OPERATOR:
      loop_control_vars.add(list(loop_cond.get_children())[0].get_definition().hash)

  return loop_control_vars


def _get_assignments(tu):
  """Return all the assignment statements found in source code, including compound assignments."""
  assignments = []
  for binop_expr in _get_nodes_by_kind(tu, [clang.CursorKind.BINARY_OPERATOR, clang.CursorKind.COMPOUND_ASSIGNMENT_OPERATOR]):
    if binop_expr.kind == clang.CursorKind.COMPOUND_ASSIGNMENT_OPERATOR:
      assignments.append(binop_expr)
    else:
      for tok in binop_expr.get_tokens():
        if tok.spelling == '=':
          assignments.append(binop_expr)

  return assignments


def _traverse_assigment_node(cursor, vars_assigned, visited, lhs):
  if cursor.kind == clang.CursorKind.INTEGER_LITERAL:
    if cursor.hash not in visited:
      vars_assigned[_get_variable_name(lhs)] = (_extract_clang(cursor))
      visited.add(cursor.hash)
      
  if cursor.kind == clang.CursorKind.BINARY_OPERATOR:
    _traverse_assigment_node(list(cursor.get_children())[0], vars_assigned, visited, lhs)
    _traverse_assigment_node(list(cursor.get_children())[1], vars_assigned, visited, lhs)
  elif cursor.kind == clang.CursorKind.PAREN_EXPR:
    _traverse_assigment_node(list(cursor.get_children())[0], vars_assigned, visited, lhs)

  visited.add(cursor.hash)


def get_relevant_constants(program_path):
  """Return the relevant constants found in source code. We consider relevants constants that appear
  on the right-hand side of assignments and that are not used in loop initialization expressions.
  """
  tu = _parse_program(program_path)
  assignments = _get_assignments(tu)
  visited = set()
  loop_vars = _get_loop_control_vars(tu)
  vars_assigned = dict()
  for a in assignments:
    lhs = list(a.get_children())[0]
    if lhs.kind == clang.CursorKind.DECL_REF_EXPR:
      # We do not consider constants used to initialize loop variables.
      if lhs.get_definition().hash in loop_vars:
        continue
    
    # If there is another assignment to the same location such that 'a' is 
    # dominant (i.e., it comes later in the code) we delete the constant
    # for we know its value will be overwritten
    lhs_name = _get_variable_name(lhs)
    if lhs_name in vars_assigned:
      del vars_assigned[lhs_name]

    for c in a.get_children():
      _traverse_assigment_node(c, vars_assigned, visited, lhs)
    
  return list(vars_assigned.values())
