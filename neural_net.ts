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

function box_muller(): number{
    const u1 = Math.random()
    const u2 = Math.random()
    const z0 = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    return z0;
}

function randn(rows: number, cols: number): number[][] {
    const arr: number[][] = []
    for(let i = 0; i < rows; i++){
        arr[i] = [];
        for(let j = 0; j < cols; j++){
            arr[i][j] = box_muller();
        }
    }
    return arr;
}

function zeroes(cols: number): number[][]{
    const arr: number[][] = [];
    for(let i = 0; i < 1; i++){
        arr[i] = [];
        for(let j = 0; j < cols; j++){
            arr[i][j] = 0;
        }
    }
    return arr;
}

let weights_input_hidden = randn(INPUT_SIZE, HIDDEN_SIZE)
    .map(row => row.map(v => v * 0.5));

let bias_hidden = zeroes(HIDDEN_SIZE);

console.log("weights_input_hidden: ", weights_input_hidden)
console.log("bias_hidden: ", bias_hidden);
