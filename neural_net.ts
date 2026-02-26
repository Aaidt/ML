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

    if(colsA !== rowsB){
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