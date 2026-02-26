const X = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
]

const Y = [
    [0],
    [1],
    [1],
    [0]
]

console.log("x: ", X, "y: ", Y)

const INPUT_SIZE = 2
const HIDDEN_SIZE = 4
const OUTPUT_SIZE = 1

function box_muller(): number {
    const u1 = Math.random()
    const u2 = Math.random()
    const z0 = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    return z0;
}

function randn(rows: number, cols: number): number[][] {
    const arr: number[][] = []
    for (let i = 0; i < rows; i++) {
        arr[i] = [];
        for (let j = 0; j < cols; j++) {
            arr[i][j] = box_muller();
        }
    }
    return arr;
}

function zeroes(cols: number): number[][] {
    const arr: number[][] = [];
    for (let i = 0; i < 1; i++) {
        arr[i] = [];
        for (let j = 0; j < cols; j++) {
            arr[i][j] = 0;
        }
    }
    return arr;
}

// Weights from input to hidden layer (2 inputs → 4 hidden neurons)
let weights_input_hidden = randn(INPUT_SIZE, HIDDEN_SIZE)
    .map(row => row.map(v => v * 0.5));
let bias_hidden = zeroes(HIDDEN_SIZE);

console.log("weights_input_hidden: ", weights_input_hidden)
console.log("bias_hidden: ", bias_hidden);

// Weights from hidden to output layer (4 hidden neurons → 1 output)
let weights_hidden_output = randn(HIDDEN_SIZE, OUTPUT_SIZE)
    .map(row => row.map(v => v * 0.5))
let bias_output = zeroes(OUTPUT_SIZE);

console.log("weights_hidden_ouput: ", weights_hidden_output)
console.log("bias_output: ", bias_output);

function sigmoid(A: number[][]): number[][] {
    return A.map(row =>
        row.map((val, i) => 1 / (1 + Math.exp(-val)))
    )
}

function sigmoid_derivative(A: number[][]): number[][] {
    const s = sigmoid(A);
    return s.map(row =>
        row.map((val) => val * (1 - val))
    )
}

function dot(A: number[][], B: number[][]): number[][] {
    const rowsA = A.length;
    const colsA = A[0].length;
    const rowsB = B.length;
    const colsB = B[0].length;

    if (colsA !== rowsB) {
        throw new Error("Matrix shap mismatch in dot()")
    }
    const arr: number[][] = Array.from({ length: rowsA }, () =>
        Array(colsB).fill(0)
    );

    for (let i = 0; i < rowsA; i++) {
        for (let j = 0; j < colsB; j++) {
            for (let k = 0; k < colsA; k++) {
                arr[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    return arr;
}

function add_bias(A: number[][], bias: number[][]): number[][] {
    return A.map(row =>
        row.map((val, i) => val += bias[0][i])
    )
}

function forward_pass(A: number[][]) {
    // Step 1: Input to Hidden
    const z_hidden = add_bias(
        dot(A, weights_input_hidden), bias_hidden
    )
    const a_hidden = sigmoid(z_hidden)

    // Step 2: Hidden to Output
    const z_output = add_bias(
        dot(a_hidden, weights_hidden_output), bias_output
    )
    const a_output = sigmoid(z_output);

    return { z_hidden, a_hidden, z_output, a_output }
}

const { z_hidden, a_hidden, z_output, a_output } = forward_pass(X);

console.log("Forward pass with untrained network: ")
console.log("-------------------------------------------------");

for (let i = 0; i < X.length; i++) {
    console.log(`Input: ${X[i]} → Prediction: ${a_output[i][0]} 
    (Target: ${Y[i][0]}))`)
}
console.log("Predictions are garbage — the network hasn't learned anything yet.")

function mean(A: number[]): number {
    return A.reduce((a, b) => a + b, 0) / A.length;
}

function flatten(A: number[][]): number[] {
    return A.reduce((flat, next) => flat.concat(next), [])
}

function compute_loss(y_true: number[][], y_pred: number[][]): number {
    const errors = y_true.map((row, i) =>
        row.map((val, j) => (val - y_pred[i][j]) ** 2)
    )
    return mean(flatten(errors));
}

const initial_loss = compute_loss(Y, a_output);
console.log(`Initial loss (untrained): ${initial_loss.toFixed(2)}`)
console.log("This number should decrease as we train")

function subtract(A: number[][], B: number[][]): number[][] {
    const rows = A.length;
    const cols = A[0].length;

    if (rows !== B.length || cols !== B[0].length) {
        throw new Error("Matrix shape mismatch in subtract()")
    }

    return A.map((row, i) =>
        row.map((val, j) => val - B[i][j])
    );
}

function transpose(A: number[][]): number[][] {
    const rows = A.length;
    const cols = A[0].length;
    const arr: number[][] = Array.from({ length: cols }, () =>
        Array(rows).fill(0)
    )
    for (let i = 0; i < rows; i++) {
        for (let j = 0; j < cols; j++) {
            arr[j][i] = A[i][j];
        }
    }
    return arr;
}

function divide_scalar(A: number[][], scalar: number): number[][] {
    return A.map(row =>
        row.map(val => val / scalar)
    )
}

function multiply_scalar(A: number[][], scalar: number): number[][] {
    return A.map(row =>
        row.map(val => val * scalar)
    )
}

function hadamard(A: number[][], B: number[][]): number[][] {
    return A.map((row, i) =>
        row.map((val, j) => val * B[i][j])
    );
}

function backward_pass(
    x: number[][],
    y: number[][],
    z_hidden: number[][],
    a_hidden: number[][],
    z_output: number[][],
    a_output: number[][],
    learning_rate: number
) {
    const m = x.length;

    // ----- OUTPUT LAYER -----
    const output_error = subtract(a_output, y);
    const output_delta = hadamard(
        output_error,
        sigmoid_derivative(z_output)
    );
    const grad_weights_hidden_output = divide_scalar(
        dot(transpose(a_hidden), output_delta),
        m
    );
    const grad_bias_output = divide_scalar(
        [flatten(output_delta)],
        m
    );

    // ----- HIDDEN LAYER -----
    const hidden_error = dot(
        output_delta,
        transpose(weights_hidden_output)
    );
    const hidden_delta = hadamard(
        hidden_error,
        sigmoid_derivative(z_hidden)
    );
    const grad_weights_input_hidden = divide_scalar(
        dot(transpose(x), hidden_delta),
        m
    );
    const grad_bias_hidden = divide_scalar(
        [flatten(hidden_delta)],
        m
    );

    // ----- UPDATE WEIGHTS -----
    weights_hidden_output = subtract(
        weights_hidden_output,
        multiply_scalar(grad_weights_hidden_output, learning_rate)
    );
    bias_output = subtract(
        bias_output,
        multiply_scalar(grad_bias_output, learning_rate)
    );
    weights_input_hidden = subtract(
        weights_input_hidden,
        multiply_scalar(grad_weights_input_hidden, learning_rate)
    );
    bias_hidden = subtract(
        bias_hidden,
        multiply_scalar(grad_bias_hidden, learning_rate)
    );
}

let learning_rate = 2.0;
let iterations = 10000;
let loss_history = [];
console.log("Training started...")
console.log("-------------------------------------------------");
for (let i = 0; i < iterations; i++) {
    const { z_hidden, a_hidden, z_output, a_output } = forward_pass(X);
    const loss = compute_loss(Y, a_output);
    loss_history.push(loss);
    backward_pass(X, Y, z_hidden, a_hidden, z_output, a_output, learning_rate)
    if (i % 2000 === 0) {
        console.log(`Iteration ${i} | Loss: ${loss.toFixed(3)}`)
    }
}

const { a_output: final_predictions } = forward_pass(X);

console.log("Final Results After Training:");
console.log("-".repeat(50));
console.log(
    "Input".padEnd(12),
    "Target".padEnd(10),
    "Prediction".padEnd(12),
    "Rounded".padEnd(10)
);
console.log("-".repeat(50));

for (let i = 0; i < X.length; i++) {
    const pred = final_predictions[i][0];
    const rounded = Math.round(pred);
    const status = rounded === Y[i][0] ? "✓" : "✗";

    console.log(
        JSON.stringify(X[i]).padEnd(12),
        String(Y[i][0]).padEnd(10),
        pred.toFixed(4).padEnd(12),
        String(rounded).padEnd(10),
        status
    );
}

console.log("-".repeat(50));
console.log("\nThe network learned XOR from random weights.");