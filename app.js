const SMALL_RESULT_LIMIT = 16;

const elements = {
  kInput: document.querySelector("#k-input"),
  nInput: document.querySelector("#n-input"),
  algorithm: document.querySelector("#algorithm-select"),
  dimensionBadge: document.querySelector("#dimension-badge"),
  manualSection: document.querySelector("#manual-section"),
  fileSection: document.querySelector("#file-section"),
  matrixA: document.querySelector("#matrix-a"),
  matrixB: document.querySelector("#matrix-b"),
  fileInput: document.querySelector("#file-input"),
  fileName: document.querySelector("#file-name"),
  randomizeButton: document.querySelector("#randomize-button"),
  runButton: document.querySelector("#run-button"),
  status: document.querySelector("#status-message"),
  resultSection: document.querySelector("#result-section"),
  resultAlgorithm: document.querySelector("#result-algorithm"),
  resultDimension: document.querySelector("#result-dimension"),
  resultTime: document.querySelector("#result-time"),
  resultGrid: document.querySelector("#result-grid"),
  resultFile: document.querySelector("#result-file"),
  resultText: document.querySelector("#result-text"),
  downloadLink: document.querySelector("#download-link")
};

let inputMode = "manual";
let loadedMatrices = null;

function getDimension() {
  const k = Number.parseInt(elements.kInput.value, 10);
  const n = Number.parseInt(elements.nInput.value, 10);
  if (!Number.isInteger(k) || k < 0) {
    throw new Error("k must be a non-negative integer.");
  }
  if (elements.nInput.value.trim() !== "" && (!Number.isInteger(n) || n < 1)) {
    throw new Error("n must be a positive integer when provided.");
  }
  return elements.nInput.value.trim() === "" ? 2 ** k : n;
}

function updateDimension() {
  try {
    const dimension = getDimension();
    elements.dimensionBadge.textContent = `Dimension: ${dimension} x ${dimension}`;
    buildMatrixInputs(dimension);
  } catch (error) {
    elements.dimensionBadge.textContent = "Dimension: check inputs";
  }
}

function buildMatrixInputs(dimension, matrices = null) {
  const build = (container, matrix) => {
    container.replaceChildren();
    container.style.gridTemplateColumns = `repeat(${dimension}, minmax(0, 1fr))`;
    for (let row = 0; row < dimension; row += 1) {
      for (let column = 0; column < dimension; column += 1) {
        const input = document.createElement("input");
        input.type = "number";
        input.step = "any";
        input.value = matrix?.[row]?.[column] ?? 0;
        input.setAttribute("aria-label", `Row ${row + 1}, column ${column + 1}`);
        container.append(input);
      }
    }
  };
  build(elements.matrixA, matrices?.[0]);
  build(elements.matrixB, matrices?.[1]);
}

function readMatrix(container, dimension) {
  const values = [...container.querySelectorAll("input")];
  if (values.length !== dimension * dimension) {
    throw new Error("The matrix input does not match the selected dimension.");
  }
  return Array.from({ length: dimension }, (_, row) =>
    values.slice(row * dimension, (row + 1) * dimension).map((input) => {
      const value = Number(input.value);
      if (!Number.isFinite(value)) {
        throw new Error("Every matrix field must contain a number.");
      }
      return value;
    })
  );
}

function zeroMatrix(size) {
  return Array.from({ length: size }, () => Array(size).fill(0));
}

function naive_mm(matA, matB) {
  const dimension = matA.length;
  const result = zeroMatrix(dimension);
  for (let row = 0; row < dimension; row += 1) {
    for (let column = 0; column < dimension; column += 1) {
      for (let index = 0; index < dimension; index += 1) {
        result[row][column] += matA[row][index] * matB[index][column];
      }
    }
  }
  return result;
}

function addMatrices(left, right) {
  return left.map((row, rowIndex) => row.map((value, columnIndex) => value + right[rowIndex][columnIndex]));
}

function subtractMatrices(left, right) {
  return left.map((row, rowIndex) => row.map((value, columnIndex) => value - right[rowIndex][columnIndex]));
}

function sliceMatrix(matrix, rowStart, rowEnd, columnStart, columnEnd) {
  return matrix.slice(rowStart, rowEnd).map((row) => row.slice(columnStart, columnEnd));
}

function strassenPowerOfTwo(matA, matB) {
  const dimension = matA.length;
  if (dimension === 1) {
    return [[matA[0][0] * matB[0][0]]];
  }
  const half = dimension / 2;
  const a11 = sliceMatrix(matA, 0, half, 0, half);
  const a12 = sliceMatrix(matA, 0, half, half, dimension);
  const a21 = sliceMatrix(matA, half, dimension, 0, half);
  const a22 = sliceMatrix(matA, half, dimension, half, dimension);
  const b11 = sliceMatrix(matB, 0, half, 0, half);
  const b12 = sliceMatrix(matB, 0, half, half, dimension);
  const b21 = sliceMatrix(matB, half, dimension, 0, half);
  const b22 = sliceMatrix(matB, half, dimension, half, dimension);

  const p = strassenPowerOfTwo(addMatrices(a11, a22), addMatrices(b11, b22));
  const q = strassenPowerOfTwo(addMatrices(a21, a22), b11);
  const r = strassenPowerOfTwo(a11, subtractMatrices(b12, b22));
  const s = strassenPowerOfTwo(a22, subtractMatrices(b21, b11));
  const t = strassenPowerOfTwo(addMatrices(a11, a12), b22);
  const u = strassenPowerOfTwo(subtractMatrices(a21, a11), addMatrices(b11, b12));
  const v = strassenPowerOfTwo(subtractMatrices(a12, a22), addMatrices(b21, b22));

  const c11 = addMatrices(subtractMatrices(addMatrices(p, s), t), v);
  const c12 = addMatrices(r, t);
  const c21 = addMatrices(q, s);
  const c22 = addMatrices(subtractMatrices(addMatrices(p, r), q), u);
  return c11.map((row, index) => row.concat(c12[index]))
    .concat(c21.map((row, index) => row.concat(c22[index])));
}

function strassens_mm(matA, matB) {
  const originalDimension = matA.length;
  let paddedDimension = 1;
  while (paddedDimension < originalDimension) paddedDimension *= 2;
  if (paddedDimension === originalDimension) return strassenPowerOfTwo(matA, matB);
  const paddedA = zeroMatrix(paddedDimension);
  const paddedB = zeroMatrix(paddedDimension);
  matA.forEach((row, index) => row.forEach((value, column) => { paddedA[index][column] = value; }));
  matB.forEach((row, index) => row.forEach((value, column) => { paddedB[index][column] = value; }));
  return strassenPowerOfTwo(paddedA, paddedB)
    .slice(0, originalDimension)
    .map((row) => row.slice(0, originalDimension));
}

function parseMatrixFile(contents) {
  const lines = contents.split(/\r?\n/).map((line) => line.trim()).filter(Boolean);
  if (!lines.length) throw new Error("The selected file is empty.");
  const dimension = Number.parseInt(lines[0], 10);
  if (!Number.isInteger(dimension) || dimension < 1) throw new Error("The first line must be a positive matrix dimension.");
  if (lines.length !== 2 * dimension + 1) throw new Error(`Expected ${2 * dimension + 1} non-empty lines, found ${lines.length}.`);
  const matrices = [1, dimension + 1].map((start) => lines.slice(start, start + dimension).map((line) => {
    const row = line.split(/\s+/).map(Number);
    if (row.length !== dimension || row.some((value) => !Number.isFinite(value))) {
      throw new Error(`Each row must contain exactly ${dimension} numbers.`);
    }
    return row;
  }));
  return { dimension, matrices };
}

function formatNumber(value) {
  return Number.isInteger(value) ? String(value) : value.toFixed(4).replace(/0+$/, "").replace(/\.$/, "");
}

function formatMatrix(matrix) {
  return `${matrix.length}\n${matrix.map((row) => row.map(formatNumber).join(" ")).join("\n")}`;
}

function showResult(result, dimension, algorithm, elapsed) {
  const resultText = formatMatrix(result);
  elements.resultSection.classList.remove("is-hidden");
  elements.resultAlgorithm.textContent = algorithm === "naive" ? "Naive" : "Strassen";
  elements.resultDimension.textContent = `${dimension} x ${dimension}`;
  elements.resultTime.textContent = `${elapsed.toFixed(3)} ms`;
  elements.resultGrid.replaceChildren();
  elements.resultGrid.style.gridTemplateColumns = `repeat(${dimension}, 58px)`;
  if (dimension <= SMALL_RESULT_LIMIT) {
    result.forEach((row) => row.forEach((value) => {
      const cell = document.createElement("span");
      cell.textContent = formatNumber(value);
      elements.resultGrid.append(cell);
    }));
    elements.resultGrid.classList.remove("is-hidden");
    elements.resultFile.classList.add("is-hidden");
  } else {
    elements.resultGrid.classList.add("is-hidden");
    elements.resultFile.classList.remove("is-hidden");
    elements.resultText.value = resultText;
  }
  elements.downloadLink.href = URL.createObjectURL(new Blob([resultText], { type: "text/plain" }));
}

function setInputMode(mode) {
  inputMode = mode;
  elements.manualSection.classList.toggle("is-hidden", mode !== "manual");
  elements.fileSection.classList.toggle("is-hidden", mode !== "file");
  document.querySelectorAll(".mode-button").forEach((button) => button.classList.toggle("is-active", button.dataset.mode === mode));
}

function randomize() {
  [elements.matrixA, elements.matrixB].forEach((container) => {
    container.querySelectorAll("input").forEach((input) => { input.value = Math.floor(Math.random() * 19) - 9; });
  });
}

elements.kInput.addEventListener("input", updateDimension);
elements.nInput.addEventListener("input", updateDimension);
elements.randomizeButton.addEventListener("click", randomize);
document.querySelectorAll(".mode-button").forEach((button) => button.addEventListener("click", () => setInputMode(button.dataset.mode)));
elements.fileInput.addEventListener("change", async () => {
  const file = elements.fileInput.files[0];
  if (!file) return;
  elements.fileName.textContent = file.name;
  try {
    loadedMatrices = parseMatrixFile(await file.text());
    elements.nInput.value = loadedMatrices.dimension;
    buildMatrixInputs(loadedMatrices.dimension, loadedMatrices.matrices);
    elements.dimensionBadge.textContent = `Dimension: ${loadedMatrices.dimension} x ${loadedMatrices.dimension}`;
    elements.status.textContent = "File loaded. Ready to run.";
  } catch (error) {
    loadedMatrices = null;
    elements.status.textContent = error.message;
  }
});
elements.runButton.addEventListener("click", () => {
  try {
    const dimension = inputMode === "file" && loadedMatrices ? loadedMatrices.dimension : getDimension();
    const matrices = inputMode === "file" ? loadedMatrices?.matrices : [readMatrix(elements.matrixA, dimension), readMatrix(elements.matrixB, dimension)];
    if (!matrices) throw new Error("Choose a valid matrix file first.");
    const start = performance.now();
    const result = elements.algorithm.value === "naive" ? naive_mm(matrices[0], matrices[1]) : strassens_mm(matrices[0], matrices[1]);
    const elapsed = performance.now() - start;
    showResult(result, dimension, elements.algorithm.value, elapsed);
    elements.status.textContent = "Complete.";
  } catch (error) {
    elements.status.textContent = error.message;
  }
});

updateDimension();
