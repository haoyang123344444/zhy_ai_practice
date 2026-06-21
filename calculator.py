def simple_calculator(a: float, b: float, op: str) -> dict:
    if op == "+":
        result = a + b
    elif op == "-":
        result = a - b
    elif op == "*":
        result = a * b
    elif op == "/":
        result = a / b
    else:
        raise ValueError(f"Unsupported operator: {op}")

    return {
        "tool": "simple_calculator",
        "a": a,
        "b": b,
        "op": op,
        "result": result,
    }