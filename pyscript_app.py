from js import Blob, URL, document, performance
from pyscript import window
from pyscript.ffi import create_proxy
import sys

sys.path.append("Matrix-multiplication")
from mm import MatrixMultiplication

SMALL_RESULT_LIMIT = 10
MANUAL_INPUT_LIMIT = 10
input_mode = "manual"
loaded_matrices = None
randomized_matrices = None
proxies = []


def element(selector):
    return document.querySelector(selector)


def dimension_from_controls():
    k = int(element("#k-input").value)
    if k < 0:
        raise ValueError("k must be a non-negative integer.")
    n_value = element("#n-input").value.strip()
    if n_value:
        n = int(n_value)
        if n < 1:
            raise ValueError("n must be a positive integer.")
        return n
    return 2 ** k


def update_dimension(event=None):
    global randomized_matrices
    randomized_matrices = None
    try:
        dimension = dimension_from_controls()
        element("#dimension-badge").textContent = f"Dimension: {dimension} x {dimension}"
        build_matrix_inputs(dimension)
    except (TypeError, ValueError):
        element("#dimension-badge").textContent = "Dimension: check inputs"


def build_matrix_inputs(dimension, matrices=None):
    too_large_for_manual_input = dimension > MANUAL_INPUT_LIMIT
    element("#manual-limit-message").classList.toggle("is-hidden", not too_large_for_manual_input)
    element("#manual-section").classList.toggle("is-hidden", input_mode != "manual" or too_large_for_manual_input)
    if too_large_for_manual_input:
        return

    for selector, matrix in (("#matrix-a", matrices[0] if matrices else None), ("#matrix-b", matrices[1] if matrices else None)):
        container = element(selector)
        container.replaceChildren()
        container.style.gridTemplateColumns = f"repeat({dimension}, minmax(0, 1fr))"
        for row in range(dimension):
            for column in range(dimension):
                input_element = document.createElement("input")
                input_element.type = "number"
                input_element.step = "any"
                input_element.value = str(matrix[row][column]) if matrix else "0"
                input_element.setAttribute("aria-label", f"Row {row + 1}, column {column + 1}")
                container.append(input_element)


def read_matrix(selector, dimension):
    values = list(element(selector).querySelectorAll("input"))
    if len(values) != dimension * dimension:
        raise ValueError("The matrix fields do not match the selected dimension.")
    matrix = []
    for row in range(dimension):
        current_row = []
        for input_element in values[row * dimension:(row + 1) * dimension]:
            try:
                current_row.append(float(input_element.value))
            except ValueError as error:
                raise ValueError("Every matrix field must contain a number.") from error
        matrix.append(current_row)
    return matrix


def parse_file(contents):
    lines = [line.strip() for line in contents.splitlines() if line.strip()]
    if not lines:
        raise ValueError("The selected file is empty.")
    dimension = int(lines[0])
    if dimension < 1 or len(lines) != 2 * dimension + 1:
        raise ValueError(f"Expected {2 * dimension + 1} non-empty lines.")
    matrices = []
    for start in (1, dimension + 1):
        matrix = []
        for line in lines[start:start + dimension]:
            row = [float(value) for value in line.split()]
            if len(row) != dimension:
                raise ValueError(f"Every row must contain {dimension} numbers.")
            matrix.append(row)
        matrices.append(matrix)
    return dimension, matrices


def format_number(value):
    value = float(value)
    return str(int(value)) if value.is_integer() else f"{value:.4f}".rstrip("0").rstrip(".")


def format_matrix(matrix):
    dimension = len(matrix)
    rows = [" ".join(format_number(value) for value in row) for row in matrix]
    return f"{dimension}\n" + "\n".join(rows)


def show_result(result, dimension, method, elapsed):
    result = result.tolist() if hasattr(result, "tolist") else result
    result_text = format_matrix(result)
    element("#result-section").classList.remove("is-hidden")
    element("#result-algorithm").textContent = method
    element("#result-dimension").textContent = f"{dimension} x {dimension}"
    element("#result-time").textContent = f"{elapsed:.3f} ms"
    grid = element("#result-grid")
    grid.replaceChildren()
    grid.style.gridTemplateColumns = f"repeat({dimension}, 58px)"
    if dimension <= SMALL_RESULT_LIMIT:
        for row in result:
            for value in row:
                cell = document.createElement("span")
                cell.textContent = format_number(value)
                grid.append(cell)
        grid.classList.remove("is-hidden")
        element("#result-file").classList.add("is-hidden")
    else:
        grid.classList.add("is-hidden")
        element("#result-file").classList.remove("is-hidden")
    blob = Blob.new([result_text], {"type": "text/plain"})
    element("#download-link").href = URL.createObjectURL(blob)


def run_multiplication(event=None):
    try:
        dimension = loaded_matrices[0] if input_mode == "file" and loaded_matrices else dimension_from_controls()
        if input_mode == "file" and loaded_matrices:
            matrices = loaded_matrices[1]
        elif randomized_matrices and randomized_matrices[0] == dimension:
            matrices = randomized_matrices[1]
        else:
            matrices = [read_matrix("#matrix-a", dimension), read_matrix("#matrix-b", dimension)]
        k = int(element("#k-input").value)
        multiplication = MatrixMultiplication(k, dim=dimension, irregular=True)
        multiplication.mat_A = matrices[0]
        multiplication.mat_B = matrices[1]
        method = element("#algorithm-select").value
        start = performance.now()
        if method == "naive":
            result = multiplication.naive_mm()
        else:
            result = multiplication.strassen_mm(dimension, matrices[0], matrices[1])
        if result is None:
            raise RuntimeError("strassen_mm returned None. Complete its even-dimension branch in mm.py.")
        elapsed = performance.now() - start
        show_result(result, dimension, "naive_mm" if method == "naive" else "strassen_mm", elapsed)
        element("#status-message").textContent = "Complete. This result came from mm.py."
    except (RuntimeError, TypeError, ValueError, IndexError) as error:
        element("#status-message").textContent = str(error)


def randomize(event=None):
    global randomized_matrices
    import random
    dimension = dimension_from_controls()
    matrices = [
        [[random.randint(-9, 9) for _ in range(dimension)] for _ in range(dimension)],
        [[random.randint(-9, 9) for _ in range(dimension)] for _ in range(dimension)],
    ]
    randomized_matrices = (dimension, matrices)
    if dimension <= MANUAL_INPUT_LIMIT:
        build_matrix_inputs(dimension, matrices)
        element("#status-message").textContent = "Random matrices generated. Ready to run mm.py."
    else:
        element("#status-message").textContent = f"Random {dimension} x {dimension} matrices generated but hidden. Ready to run mm.py."


def set_mode(mode):
    global input_mode
    input_mode = mode
    dimension = dimension_from_controls()
    element("#manual-section").classList.toggle("is-hidden", mode != "manual" or dimension > MANUAL_INPUT_LIMIT)
    element("#file-section").classList.toggle("is-hidden", mode != "file")
    element("#manual-mode").classList.toggle("is-active", mode == "manual")
    element("#file-mode").classList.toggle("is-active", mode == "file")


async def on_file_change(event):
    global loaded_matrices
    file = event.target.files.item(0)
    if not file:
        return
    element("#file-name").textContent = file.name
    try:
        dimension, matrices = parse_file(await file.text())
        loaded_matrices = (dimension, matrices)
        element("#n-input").value = str(dimension)
        build_matrix_inputs(dimension, matrices)
        element("#dimension-badge").textContent = f"Dimension: {dimension} x {dimension}"
        element("#status-message").textContent = "File loaded. Ready to run mm.py."
    except (TypeError, ValueError) as error:
        loaded_matrices = None
        element("#status-message").textContent = str(error)


def bind(selector, event_name, callback):
    proxy = create_proxy(callback)
    proxies.append(proxy)
    element(selector).addEventListener(event_name, proxy)


bind("#k-input", "input", update_dimension)
bind("#n-input", "input", update_dimension)
bind("#run-button", "click", run_multiplication)
bind("#randomize-button", "click", randomize)
bind("#manual-mode", "click", lambda event: set_mode("manual"))
bind("#file-mode", "click", lambda event: set_mode("file"))
bind("#file-input", "change", on_file_change)
update_dimension()
element("#status-message").textContent = "Python loaded. Ready for input."
