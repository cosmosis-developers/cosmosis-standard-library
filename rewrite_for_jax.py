import ast
import inspect
import copy
import types
import builtins

def _make_cell(value):
    # Create a closure cell containing value.
    # Using a nested function so we can grab its closure cell.
    def _inner():
        return value
    return _inner.__closure__[0]

def _closure_from_mapping(freevar_names, name_to_value, original_closure):
    """
    Build a tuple of cell objects ordered to match freevar_names.
    name_to_value: mapping from name -> value (typically from original closure or globals)
    original_closure: original function.__closure__ (may be None)
    """
    cells = []
    # try to build mapping from original closure first if available
    orig_map = {}
    if original_closure and hasattr(original_closure, "__len__"):
        # original function's freevar names are not available here; caller should map them
        # so original_closure is used only for extracting cell_contents if provided mapping
        pass
    for name in freevar_names:
        if name in name_to_value:
            cells.append(_make_cell(name_to_value[name]))
        else:
            # fallback to None
            cells.append(_make_cell(None))
    return tuple(cells)

def abstract_syntax_tree_rewriter(function):
    """
    Rewrites the AST of `function`, compiles it, and returns a new function object
    that preserves the original function's globals and closure values where possible.

    Returns (new_function, source_text)
    """
    src = inspect.getsource(function)
    # If the function is indented (e.g., defined in a class or nested),
    # dedent so ast.parse works as expected.
    # src = inspect.cleandoc(src)
    tree = ast.parse(src)

    class Rewriter(ast.NodeTransformer):
        def visit_Return(self, node):
            # keep original value for inspection
            val = node.value
            if val is None:
                return node

            # if already returning the block -> do nothing
            if isinstance(val, ast.Name) and val.id == "block":
                return node

            # literal numeric return
            if isinstance(val, ast.Constant) and isinstance(val.value, (int, float)):
                if val.value == 0:
                    # return block
                    new_node = ast.Return(value=ast.Name(id="block", ctx=ast.Load()))
                    return ast.copy_location(new_node, node)
                else:
                    # replace with raising an error
                    raise_call = ast.Raise(
                        exc=ast.Call(
                            func=ast.Name(id="RuntimeError", ctx=ast.Load()),
                            args=[ast.Constant("Error in module")],
                            keywords=[]
                        ),
                        cause=None
                    )
                    return ast.copy_location(raise_call, node)

            # non-literal return: replace with if val == 0: return block else: raise
            cmp_val = copy.deepcopy(val)
            compare = ast.Compare(left=cmp_val, ops=[ast.Eq()], comparators=[ast.Constant(0)])
            if_body = [ast.Return(value=ast.Name(id="block", ctx=ast.Load()))]
            else_body = [
                ast.Raise(
                    exc=ast.Call(
                        func=ast.Name(id="RuntimeError", ctx=ast.Load()),
                        args=[ast.Constant("Error in module")],
                        keywords=[]
                    ),
                    cause=None
                )
            ]
            if_node = ast.If(test=compare, body=if_body, orelse=else_body)
            return ast.copy_location(if_node, node)

    new_tree = Rewriter().visit(tree)
    ast.fix_missing_locations(new_tree)

    # We'll exec the rewritten code in a fresh namespace that uses the original
    # function's globals as its base. This preserves imports and global names.
    original_globals = function.__globals__
    exec_namespace = dict(original_globals)  # copy to avoid mutating original globals unintentionally

    # Compile and exec the module AST
    try:
        compiled = compile(new_tree, filename="<ast>", mode="exec")
    except:
        source = ast.unparse(new_tree)
        print(source)
        return None
    exec(compiled, exec_namespace)

    # Get the rewritten function object from the namespace by name
    new_candidate = exec_namespace.get(function.__name__)
    if new_candidate is None or not isinstance(new_candidate, types.FunctionType):
        # The AST might contain multiple definitions or the function may be nested.
        # Try to find a function with same qualname.
        for obj in exec_namespace.values():
            if isinstance(obj, types.FunctionType) and obj.__name__ == function.__name__:
                new_candidate = obj
                break
    if new_candidate is None:
        raise RuntimeError("Could not find rewritten function in exec namespace.")

    # Now re-create the function to preserve the original function's globals and closure.
    # We need to match the freevars expected by the new code object.
    new_code = new_candidate.__code__
    new_freevars = new_code.co_freevars  # tuple of names that must be provided via closure

    # Build mapping of freevar name -> value from original function's closure and globals.
    name_to_value = {}
    # original freevars/names -> values from original function's closure
    if function.__closure__ and function.__code__.co_freevars:
        for name, cell in zip(function.__code__.co_freevars, function.__closure__):
            try:
                name_to_value[name] = cell.cell_contents
            except ValueError:
                # cell empty; ignore
                name_to_value[name] = None

    # also fall back to globals in case a freevar is provided from a global
    for k, v in original_globals.items():
        if k not in name_to_value:
            name_to_value[k] = v

    # Create closure cells matching new_freevars order
    new_closure = _closure_from_mapping(new_freevars, name_to_value, function.__closure__) if new_freevars else None

    # Build the new function object
    new_func = types.FunctionType(
        new_code,
        original_globals,
        name=function.__name__,
        argdefs=function.__defaults__,
        closure=new_closure
    )
    # copy kwdefaults, annotations, and dict
    new_func.__kwdefaults__ = getattr(function, "__kwdefaults__", None)
    new_func.__annotations__ = getattr(function, "__annotations__", {}).copy()
    new_func.__dict__.update(getattr(function, "__dict__", {}))
    new_func.__module__ = getattr(function, "__module__", function.__globals__.get("__name__"))

    # Produce source code string for the rewritten function (optional)
    try:
        source = ast.unparse(new_tree)
    except Exception:
        source = None

    return new_func, source