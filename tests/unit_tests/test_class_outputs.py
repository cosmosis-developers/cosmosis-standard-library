import ast
from pathlib import Path


CLASS_SOURCE = Path(__file__).parents[2] / "boltzmann/class/class_interface.py"


def test_nonlinear_only_class_output_allocates_its_grid():
    tree = ast.parse(CLASS_SOURCE.read_text())
    nonlinear_branches = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        test = ast.unparse(node.test)
        if test == "c.nonlinear_method != 0":
            nonlinear_branches.extend(node.body)
    assignments = [node for node in nonlinear_branches
                   if isinstance(node, ast.Assign) and
                   any(isinstance(target, ast.Name) and target.id == "P"
                       for target in node.targets)]
    assert assignments, "nonlinear-only CLASS output must allocate P"
