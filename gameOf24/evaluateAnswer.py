import ast
from collections import Counter

# evaluateAnswer.py
# Evaluates whether a proposed solution to a "Game of 24" puzzle is valid.
# Call evaluate_answer(puzzle: str, proposedSolution: str)

_ALLOWED_BINOPS = (ast.Add, ast.Sub, ast.Mult, ast.Div)
_ALLOWED_UNARYOPS = (ast.UAdd, ast.USub)
_TOL = 1e-9

def _safe_eval(node):
    """Evaluate an AST node allowing only +, -, *, / and numeric constants."""
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.BinOp):
        if not isinstance(node.op, _ALLOWED_BINOPS):
            raise ValueError("operator not allowed")
        left = _safe_eval(node.left)
        right = _safe_eval(node.right)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
    if isinstance(node, ast.UnaryOp):
        if not isinstance(node.op, _ALLOWED_UNARYOPS):
            raise ValueError("unary operator not allowed")
        val = _safe_eval(node.operand)
        return +val if isinstance(node.op, ast.UAdd) else -val
    if isinstance(node, ast.Constant):  # Python 3.8+
        if isinstance(node.value, (int, float)):
            return float(node.value)
        raise ValueError("only numeric constants allowed")
    # older ast.Num support (for very old Python versions)
    if hasattr(ast, "Num") and isinstance(node, ast.Num):
        return float(node.n)
    raise ValueError("disallowed expression element: %r" % type(node))

def _collect_constants(node, out):
    """Collect numeric constants (as floats) from AST node into out list."""
    if isinstance(node, ast.Expression):
        _collect_constants(node.body, out); return
    if isinstance(node, ast.BinOp):
        _collect_constants(node.left, out)
        _collect_constants(node.right, out)
        return
    if isinstance(node, ast.UnaryOp):
        # unary op may modify sign; evaluate operand and apply sign to constant if possible
        if isinstance(node.operand, (ast.Constant, getattr(ast, "Num", None))):
            # evaluate this small subtree safely
            try:
                val = _safe_eval(node)
            except Exception:
                pass
            else:
                out.append(val)
                return
        _collect_constants(node.operand, out)
        return
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            out.append(float(node.value))
        return
    if hasattr(ast, "Num") and isinstance(node, ast.Num):
        out.append(float(node.n))
        return
    # recurse into other possible container nodes (just in case)
    for child in ast.iter_child_nodes(node):
        _collect_constants(child, out)

def _parse_expr(expr):
    """Parse expr (string) into AST Expression node, disallowing names/calls/etc."""
    parsed = ast.parse(expr, mode="eval")
    # validate AST contains only allowed node types
    for node in ast.walk(parsed):
        if isinstance(node, ast.Call) or isinstance(node, ast.Name) or isinstance(node, ast.Attribute):
            raise ValueError("disallowed construct in expression")
    return parsed

def evaluate_answer(puzzle: str, proposedSolution: str) -> bool:
    """
    puzzle: string like '1 1 4 6'
    proposedSolution: expression or equation string that should use each puzzle number exactly once
    Returns True if proposedSolution is valid, uses each puzzle number exactly once, and equals 24.
    """
    # parse puzzle numbers
    try:
        puzzle_tokens = puzzle.strip().split()
        if len(puzzle_tokens) != 4:
            return False
        puzzle_nums = [float(int(tok)) if tok.lstrip("-").isdigit() else float(tok) for tok in puzzle_tokens]
    except Exception:
        return False

    # split equation if '=' present
    parts = proposedSolution.split('=')
    if len(parts) > 2:
        return False
    try:
        if len(parts) == 2:
            left_s, right_s = parts[0].strip(), parts[1].strip()
            left_ast = _parse_expr(left_s)
            right_ast = _parse_expr(right_s)
            left_val = _safe_eval(left_ast)
            right_val = _safe_eval(right_ast)
            # both sides must evaluate equal
            if abs(left_val - right_val) > 1e-6:
                return False
            result_val = left_val  # common value
            combined_asts = [left_ast, right_ast]
        else:
            expr_s = proposedSolution.strip()
            expr_ast = _parse_expr(expr_s)
            result_val = _safe_eval(expr_ast)
            combined_asts = [expr_ast]
    except Exception:
        return False

    # result must equal 24
    if abs(result_val - 24.0) > 1e-6:
        return False

    # collect numeric constants from the whole equation/expression
    constants = []
    for a in combined_asts:
        _collect_constants(a, constants)

    # Count occurrences of puzzle numbers and ensure they appear exactly as in puzzle
    puzzle_count = Counter(puzzle_nums)
    # For each puzzle number, count how many constants equal it (within tol)
    found_count = Counter()
    for pnum in puzzle_count:
        cnt = sum(1 for c in constants if abs(c - pnum) < _TOL)
        found_count[pnum] = cnt

    if puzzle_count != found_count:
        return False

    return True