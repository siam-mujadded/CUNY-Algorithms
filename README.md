# Matrix Lab

An interactive GitHub Pages frontend for comparing naive matrix multiplication with Strassen multiplication. It runs entirely in the browser, so matrix data is never uploaded.

## Test the Python code in the browser

Open [`pyscript.html`](pyscript.html) to use the Python-compatible version. It uses [PyScript](https://pyscript.net/) and Pyodide to load `Matrix-multiplication/mm.py` and NumPy in the browser. The page calls the actual `MatrixMultiplication.naive_mm` and `MatrixMultiplication.strassen_mm` methods, so this is the best page for testing changes to the Python implementation.

PyScript downloads its runtime and NumPy from the internet on first load. GitHub Pages can serve this page as-is; there is no build step or backend. Do not open `pyscript.html` directly with a `file://` URL because browsers block PyScript from fetching `mm.py` that way.

## Open it on GitHub Pages

1. Push this repository to GitHub.
2. Open **Settings > Pages** for the repository.
3. Under **Build and deployment**, choose **Deploy from a branch**.
4. Select the branch containing these files and the `/ (root)` folder.
5. Open the published URL shown by GitHub.

The Python test page will be available at `https://<username>.github.io/<repository>/pyscript.html`.

The site is static: GitHub Pages does not need a build command, server, package manager, or backend. `index.html`, `style.css`, and `app.js` must remain together in the published folder.

## Run locally

Run a local HTTP server from the repository root. Opening `pyscript.html` directly with a `file://` URL will cause a CORS error when PyScript tries to load `mm.py`.

```text
python -m http.server 8000
```

Then visit `http://localhost:8000/pyscript.html`.

## Using the app

- `k` is required. If `n` is empty, the matrix dimension is `2^k`.
- `n` is optional and overrides `2^k`, allowing arbitrary square dimensions.
- Choose **Naive multiplication** or **Strassen multiplication**.
- Use **Enter matrices** to fill both matrices in the page, or use **Load a file**.
- The file format is `2n + 1` non-empty lines: the first line is `n`, followed by `n` rows for matrix A and `n` rows for matrix B.

Example `2 x 2` input file:

```text
2
1 2
3 4
5 6
7 8
```

The elapsed time measures only the selected multiplication function. Results up to `10 x 10` are shown as a grid. Larger results are hidden in the page and can be viewed by downloading the TXT file with **Download result**.

## Relationship to `mm.py`

The default `index.html` uses browser-compatible implementations in `app.js`. For testing the actual Python class, use `pyscript.html`; it imports `mm.py` directly and calls `naive_mm` or `strassen_mm`.
