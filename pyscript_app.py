from js import Blob, URL, document, performance
from pyscript import window
from pyscript.ffi import create_proxy
import asyncio
import sys
import numpy as np
import random

sys.path.append("Matrix-multiplication")
from mm import MatrixMultiplication

SMALL_RESULT_LIMIT = 10
MANUAL_INPUT_LIMIT = 10
RANDOM_VALUE_LIMIT = 9
input_mode = "manual"
loaded_matrices = None
randomized_matrices = None
benchmark_cases = None
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


def download_text(selector, filename, contents):
    blob = Blob.new([contents], {"type": "text/plain"})
    link = element(selector)
    link.href = URL.createObjectURL(blob)
    link.download = filename
    link.classList.remove("is-hidden")


def parse_benchmark_file(contents):
    lines = [line.strip() for line in contents.splitlines() if line.strip()]
    example = "Example: 1\\n2\\n1 0\\n0 1\\n1 2\\n3 4"
    if not lines:
        raise ValueError(f"Benchmark file is empty. {example}")
    try:
        case_count = int(lines[0])
    except ValueError as error:
        raise ValueError(f"The first line must be the number of cases, from 1 to 7. {example}") from error
    if not 1 <= case_count <= 7:
        raise ValueError(f"The first line must be the number of cases, from 1 to 7. {example}")

    cases = []
    line_index = 1
    for case_number in range(1, case_count + 1):
        if line_index >= len(lines):
            raise ValueError(f"Missing k for case {case_number}. {example}")
        try:
            exponent = int(lines[line_index])
        except ValueError as error:
            raise ValueError(f"Case {case_number} must start with an integer k from 1 to 7. {example}") from error
        line_index += 1
        if not 1 <= exponent <= 7:
            raise ValueError(f"Case {case_number} has invalid k. Use a value from 1 to 7. {example}")
        dimension = 2 ** exponent
        matrices = []
        for matrix_number in range(2):
            matrix = []
            for row_number in range(dimension):
                if line_index >= len(lines):
                    raise ValueError(f"Case {case_number} is missing matrix rows. {example}")
                row = lines[line_index].split()
                line_index += 1
                if len(row) != dimension:
                    raise ValueError(f"Case {case_number} row {row_number + 1} must contain {dimension} values. {example}")
                try:
                    matrix.append([float(value) for value in row])
                except ValueError as error:
                    raise ValueError(f"Case {case_number} contains a non-numeric value. {example}") from error
            matrices.append(matrix)
        cases.append((exponent, dimension, matrices))
    if line_index != len(lines):
        raise ValueError(f"The file has extra rows after the declared cases. {example}")
    return cases


def benchmark_file_text(cases):
    lines = [str(len(cases))]
    for exponent, _, matrices in cases:
        lines.append(str(exponent))
        lines.extend(" ".join(format_number(value) for value in row) for matrix in matrices for row in matrix)
    return "\n".join(lines)


def measured_operation_rows(max_exponent):
    rows = []
    for exponent in range(max_exponent + 1):
        dimension = 2 ** exponent
        ones = [[1] * dimension for _ in range(dimension)]
        multiplication = MatrixMultiplication(exponent, dim=dimension, irregular=False)
        multiplication.mat_A = ones
        multiplication.mat_B = ones
        _, naive_additions, naive_multiplications = multiplication.naive_mm()
        _, strassen_additions, strassen_multiplications = multiplication.strassen_mm(dimension, ones, ones)
        rows.append({
            "k": exponent,
            "dimension": dimension,
            "naive_additions": naive_additions,
            "naive_multiplications": naive_multiplications,
            "naive_total": naive_additions + naive_multiplications,
            "strassen_additions": strassen_additions,
            "strassen_multiplications": strassen_multiplications,
            "strassen_total": strassen_additions + strassen_multiplications,
        })
    return rows


def operation_csv(rows):
    lines = ["k,n,naive_additions_subtractions,naive_multiplications,naive_total,strassen_additions_subtractions,strassen_multiplications,strassen_total"]
    for row in rows:
        lines.append(",".join(str(row[key]) for key in (
            "k", "dimension", "naive_additions", "naive_multiplications", "naive_total",
            "strassen_additions", "strassen_multiplications", "strassen_total")))
    return "\n".join(lines)


def timing_csv(rows):
    lines = ["k,n,naive_time_ms,strassen_time_ms"]
    for row in rows:
        lines.append(f"{row['k']},{row['dimension']},{row['naive_ms']:.6f},{row['strassen_ms']:.6f}")
    return "\n".join(lines)


def generate_benchmark(event=None):
    global benchmark_cases
    case_count = int(element("#benchmark-case-count").value) if element("#benchmark-case-count") else 7
    cases = []
    for exponent in range(1, case_count + 1):
        dimension = 2 ** exponent
        matrices = [
            [[random.randint(-5, 5) for _ in range(dimension)] for _ in range(dimension)],
            [[random.randint(-5, 5) for _ in range(dimension)] for _ in range(dimension)],
        ]
        cases.append((exponent, dimension, matrices))
    benchmark_cases = cases
    download_text("#benchmark-download-link", "matrix-benchmark.txt", benchmark_file_text(cases))
    element("#benchmark-status").textContent = f"Generated {case_count} test cases. Download the file or run it now."


def set_busy(is_busy):
    element("#spinner").classList.toggle("is-hidden", not is_busy)
    element("#run-button").disabled = is_busy
    element("#run-benchmark-button").disabled = is_busy


def verify_result(result, matrix_a, matrix_b):
    expected = np.matmul(np.asarray(matrix_a), np.asarray(matrix_b))
    return bool(np.allclose(np.asarray(result), expected))


def render_line_chart(selector, rows, first_key, second_key, first_label, second_label, y_label, logarithmic=False):
    chart = element(selector)
    chart.replaceChildren()
    if not rows:
        return

    namespace = "http://www.w3.org/2000/svg"
    width, height = 820, 430
    margin_left, margin_right, margin_top, margin_bottom = 70, 30, 45, 65
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    maximum = max(max(row[first_key], row[second_key]) for row in rows) or 1
    y_limit = maximum * 1.1
    log_limit = np.log10(max(y_limit, 10)) if logarithmic else None

    svg = document.createElementNS(namespace, "svg")
    svg.setAttribute("viewBox", f"0 0 {width} {height}")
    svg.setAttribute("role", "img")
    svg.setAttribute("aria-label", "Naive and Strassen elapsed time by k")

    def svg_element(tag, attributes=None, text=None):
        node = document.createElementNS(namespace, tag)
        for key, value in (attributes or {}).items():
            node.setAttribute(key, str(value))
        if text is not None:
            node.textContent = str(text)
        svg.append(node)
        return node

    x_axis = margin_left
    y_axis = height - margin_bottom
    svg_element("line", {"x1": x_axis, "y1": margin_top, "x2": x_axis, "y2": y_axis, "class": "chart-axis"})
    svg_element("line", {"x1": x_axis, "y1": y_axis, "x2": width - margin_right, "y2": y_axis, "class": "chart-axis"})
    svg_element("text", {"x": width / 2, "y": height - 15, "class": "chart-axis-label", "text-anchor": "middle"}, "k (matrix exponent)")
    svg_element("text", {"x": 17, "y": height / 2, "class": "chart-axis-label", "text-anchor": "middle", "transform": f"rotate(-90 17 {height / 2})"}, y_label)

    if logarithmic:
        tick_values = [10 ** exponent for exponent in range(int(np.floor(np.log10(maximum))) + 1) if 10 ** exponent <= y_limit]
        if 1 not in tick_values:
            tick_values.insert(0, 1)
    else:
        tick_values = [y_limit * tick / 5 for tick in range(6)]
    for value in tick_values:
        fraction = np.log10(max(value, 1)) / log_limit if logarithmic else value / y_limit
        y = y_axis - plot_height * fraction
        svg_element("line", {"x1": x_axis, "y1": y, "x2": width - margin_right, "y2": y, "class": "chart-grid-line"})
        svg_element("text", {"x": margin_left - 10, "y": y + 4, "class": "chart-tick", "text-anchor": "end"}, f"{value:.1f}")

    def point(index, value):
        x = x_axis if len(rows) == 1 else x_axis + plot_width * index / (len(rows) - 1)
        fraction = np.log10(max(value, 1)) / log_limit if logarithmic else value / y_limit
        y = y_axis - plot_height * fraction
        return x, y

    for index, row in enumerate(rows):
        x, _ = point(index, 0)
        svg_element("line", {"x1": x, "y1": y_axis, "x2": x, "y2": y_axis + 5, "class": "chart-axis"})
        svg_element("text", {"x": x, "y": y_axis + 22, "class": "chart-tick", "text-anchor": "middle"}, row["k"])

    for key, color, label in ((first_key, "#df5d35", first_label), (second_key, "#202a25", second_label)):
        points = " ".join(f"{x},{y}" for x, y in (point(index, row[key]) for index, row in enumerate(rows)))
        svg_element("polyline", {"points": points, "class": "chart-line", "stroke": color})
        for index, row in enumerate(rows):
            x, y = point(index, row[key])
            circle = svg_element("circle", {"cx": x, "cy": y, "r": 5, "class": "chart-point", "fill": color})
            circle.setAttribute("title", f"{label}, k={row['k']}: {row[key]:.3f} ms")

    legend_x = width - 190
    for index, (color, label) in enumerate((("#df5d35", first_label), ("#202a25", second_label))):
        y = margin_top - 20 + index * 22
        svg_element("line", {"x1": legend_x, "y1": y, "x2": legend_x + 24, "y2": y, "stroke": color, "class": "chart-line"})
        svg_element("text", {"x": legend_x + 32, "y": y + 4, "class": "chart-legend"}, label)

    chart.append(svg)
    chart.classList.remove("is-hidden")


def render_benchmark_chart(rows):
    render_line_chart("#benchmark-chart", rows, "naive_ms", "strassen_ms", "Naive", "Strassen", "Time (ms)")


def render_operation_charts(rows):
    render_line_chart("#addition-chart", rows, "naive_additions", "strassen_additions", "Naive", "Strassen", "Additions / subtractions", logarithmic=True)
    render_line_chart("#multiplication-chart", rows, "naive_multiplications", "strassen_multiplications", "Naive", "Strassen", "Multiplications", logarithmic=True)
    render_line_chart("#operation-total-chart", rows, "naive_total", "strassen_total", "Naive", "Strassen", "Total operations", logarithmic=True)


def render_table(selector, headers, rows):
    container = element(selector)
    container.replaceChildren()
    table = document.createElement("table")
    header_row = document.createElement("tr")
    for header in headers:
        cell = document.createElement("th")
        cell.textContent = header
        header_row.append(cell)
    table.append(header_row)
    for values in rows:
        table_row = document.createElement("tr")
        for value in values:
            cell = document.createElement("td")
            cell.textContent = str(value)
            table_row.append(cell)
        table.append(table_row)
    container.append(table)
    container.classList.remove("is-hidden")


def render_evaluation_tables(timing_rows, operation_data):
    render_table(
        "#timing-table",
        ("k", "n", "Naive time (ms)", "Strassen time (ms)"),
        ((row["k"], row["dimension"], f"{row['naive_ms']:.6f}", f"{row['strassen_ms']:.6f}") for row in timing_rows),
    )
    render_table(
        "#operation-table",
        ("n", "Naive add/sub", "Naive mult", "Naive total", "Strassen add/sub", "Strassen mult", "Strassen total"),
        ((row["dimension"], row["naive_additions"], row["naive_multiplications"], row["naive_total"], row["strassen_additions"], row["strassen_multiplications"], row["strassen_total"]) for row in operation_data),
    )


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


async def run_multiplication(event=None):
    set_busy(True)
    try:
        await asyncio.sleep(0)
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
            result, additions, multiplications = multiplication.naive_mm()
        else:
            result, additions, multiplications = multiplication.strassen_mm(dimension, matrices[0], matrices[1], arbitrary_dim=True)
        if result is None:
            raise RuntimeError("strassen_mm returned None. Complete its even-dimension branch in mm.py.")
        elapsed = performance.now() - start
        show_result(result, dimension, "naive_mm" if method == "naive" else "strassen_mm", elapsed)
        element("#status-message").textContent = f"Complete. {additions} additions/subtractions and {multiplications} multiplications counted."
    except (RuntimeError, TypeError, ValueError, IndexError) as error:
        element("#status-message").textContent = str(error)
    finally:
        set_busy(False)


async def run_benchmark(event=None):
    global benchmark_cases
    set_busy(True)
    element("#benchmark-status").textContent = "Running benchmark..."
    try:
        await asyncio.sleep(0)
        if not benchmark_cases:
            raise ValueError("Choose a benchmark file or generate one first.")
        check_correctness = element("#correctness-check").checked
        rows = []
        report = ["Matrix multiplication benchmark", "k,dimension,naive_ms,strassen_ms,naive_total_operations,strassen_total_operations,naive_correct,strassen_correct"]
        for exponent, dimension, matrices in benchmark_cases:
            multiplication = MatrixMultiplication(exponent, dim=dimension, irregular=False)
            multiplication.mat_A = matrices[0]
            multiplication.mat_B = matrices[1]

            start = performance.now()
            naive_result, naive_additions, naive_multiplications = multiplication.naive_mm()
            naive_elapsed = performance.now() - start

            start = performance.now()
            strassen_result, strassen_additions, strassen_multiplications = multiplication.strassen_mm(dimension, matrices[0], matrices[1])
            strassen_elapsed = performance.now() - start
            naive_correct = verify_result(naive_result, matrices[0], matrices[1]) if check_correctness else "not checked"
            strassen_correct = verify_result(strassen_result, matrices[0], matrices[1]) if check_correctness else "not checked"
            rows.append({"k": exponent, "dimension": dimension, "naive_ms": naive_elapsed, "strassen_ms": strassen_elapsed})
            report.append(f"{exponent},{dimension},{naive_elapsed:.6f},{strassen_elapsed:.6f},{naive_additions + naive_multiplications},{strassen_additions + strassen_multiplications},{naive_correct},{strassen_correct}")
            report.append("Naive result:")
            report.append(format_matrix(naive_result))
            report.append("Strassen result:")
            report.append(format_matrix(strassen_result))
            await asyncio.sleep(0)

        operation_data = measured_operation_rows(max(row["k"] for row in rows))
        download_text("#operation-csv-link", "operation-counts.csv", operation_csv(operation_data))
        download_text("#timing-csv-link", "timings.csv", timing_csv(rows))
        render_benchmark_chart(rows)
        render_operation_charts(operation_data)
        render_evaluation_tables(rows, operation_data)
        download_text("#benchmark-result-link", "matrix-results.txt", "\n".join(report))
        element("#benchmark-status").textContent = f"Completed {len(rows)} cases. Download the result file for timings and verification."
    except (RuntimeError, TypeError, ValueError, IndexError) as error:
        element("#benchmark-status").textContent = str(error)
    finally:
        set_busy(False)


def randomize(event=None):
    global randomized_matrices
    import random
    dimension = dimension_from_controls()
    matrices = [
        [[random.randint(-RANDOM_VALUE_LIMIT, RANDOM_VALUE_LIMIT) for _ in range(dimension)] for _ in range(dimension)],
        [[random.randint(-RANDOM_VALUE_LIMIT, RANDOM_VALUE_LIMIT) for _ in range(dimension)] for _ in range(dimension)],
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


async def on_benchmark_file_change(event):
    global benchmark_cases
    file = event.target.files.item(0)
    if not file:
        return
    element("#benchmark-file-name").textContent = file.name
    try:
        benchmark_cases = parse_benchmark_file(await file.text())
        element("#benchmark-status").textContent = f"Validated {len(benchmark_cases)} test cases. Ready to run."
    except (TypeError, ValueError) as error:
        benchmark_cases = None
        element("#benchmark-status").textContent = str(error)


def bind(selector, event_name, callback):
    proxy = create_proxy(callback)
    proxies.append(proxy)
    element(selector).addEventListener(event_name, proxy)


bind("#k-input", "input", update_dimension)
bind("#n-input", "input", update_dimension)
bind("#run-button", "click", run_multiplication)
bind("#run-benchmark-button", "click", run_benchmark)
bind("#generate-benchmark-button", "click", generate_benchmark)
bind("#randomize-button", "click", randomize)
bind("#manual-mode", "click", lambda event: set_mode("manual"))
bind("#file-mode", "click", lambda event: set_mode("file"))
bind("#file-input", "change", on_file_change)
bind("#benchmark-file-input", "change", on_benchmark_file_change)
update_dimension()
element("#status-message").textContent = "Python loaded. Ready for input."
