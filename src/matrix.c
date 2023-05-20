#include  <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include "defines.h"

// typedef double double;


// TODO:: Change double to double via macro

// Creates a matrix of order row x col
matrix matrix_create(int row, int col) {
    matrix m;
    m.row = row;
    m.col = col;
    m.values = malloc(row * sizeof(double*));

    for(int i = 0; i < row; i++)
        m.values[i] = calloc(col, sizeof(double));

    return m;
}

// Frees the memory allocated to the matrix
void matrix_free(matrix m) {
    for(int i = 0; i < m.row; i++)
        free(m.values[i]);
    free(m.values);
}

// Copies in to out
void matrix_copy(matrix in, matrix out) {
    if(in.row != out.row || in.col != out.col) {
        printf("Incorrect matrix dimensions [copy]\n");
        printf("[%d %d] [%d %d]\n", in.row, in.col, out.row, out.col);
        exit(1);
    }

    for(int i = 0; i < in.row; i++) {
        for(int j = 0; j < in.col; j++)
            out.values[i][j] = in.values[i][j];
    }
}

// Prints the matrix
void matrix_print(matrix a, char *msg) {
    printf("%s\tR: %d\tC: %d\n", msg, a.row, a.col);
    for(int i = 0; i < a.row; i++) {
        for(int j = 0; j < a.col; j++) 
            printf("%f ", a.values[i][j]);
        printf("\n");
    }       
    printf("\n");
}

void matrix_fill(matrix in, double value) {
    for(int i = 0; i < in.row; i++) 
        for(int j = 0; j < in.col; j++)
            in.values[i][j] = value; 
    // bzero()
    
    // memset(in.values, 0, sizeof(in.values[0][0]) * in.row * in.col);
}

// Adds matrix a with b and stores it in c
void matrix_add(matrix a, matrix b, matrix c) {
    if(a.row != b.row || b.row != c.row || a.col != b.col || b.col != c.col) {
        printf("Incorrect matrix dimensions [addtion]\n");
        printf("[%d %d] [%d %d] [%d %d]\n", a.row, a.col, b.row, b.col, c.row, c.col);
        exit(1);
    }
    if(a.values == NULL || b.values == NULL || c.values == NULL) {
        printf("Matrices shoulb not be null [addition]");
        exit(1);
    }

    for(int i = 0; i < a.row; i++) {
        for(int j = 0; j < a.col; j++)
            c.values[i][j] = a.values[i][j] + b.values[i][j];
    }
}

// Subtracts matrix a with b and stores it in c
void matrix_sub(matrix a, matrix b, matrix c) {
    if(a.row != b.row || b.row != c.row || a.col != b.col || b.col != c.col) {
        printf("Incorrect matrix dimensions [subtraction]\n");
        printf("[%d %d] [%d %d] [%d %d]\n", a.row, a.col, b.row, b.col, c.row, c.col);
        exit(1);
    }
    if(a.values == NULL || b.values == NULL || c.values == NULL) {
        printf("Matrices shoulb not be null [subtraction]");
        exit(1);
    }

    for(int i = 0; i < a.row; i++) {
        for(int j = 0; j < a.col; j++)
            c.values[i][j] = a.values[i][j] - b.values[i][j];
    }
}

// Multiply all elements of matrix a with given scalar
void matrix_scalar_multiply(matrix a, double scalar, matrix b) {
    if(a.values == NULL) {
        printf("Matrices should not be null [scalar_multiplication]");
        exit(1);
    }

    for(int i = 0; i < a.row; i++) {
        for(int j = 0; j < a.col; j++)
            b.values[i][j] = a.values[i][j] * scalar;
    }
}

// Multiplies matrix a with b and stores it in c
void matrix_multiply(matrix a, matrix b, matrix c) {
    if(a.col != b.row || a.row != c.row || c.col != b.col) {
        printf("Incorrect matrix dimensions [matrix_multiply]\n");
        printf("[%d %d] [%d %d] [%d %d]\n", a.row, a.col, b.row, b.col, c.row, c.col);
        exit(1);
    }
    if(a.values == NULL || b.values == NULL || c.values == NULL) {
        printf("Matrices should not be null [matrix_multiply]");
        exit(1);
    }

    for(int i = 0; i < a.row; i++) {
        for(int j = 0; j < b.col; j++) {
            c.values[i][j] = 0;
            for (int k = 0; k < a.col; k++)
                c.values[i][j] += a.values[i][k] * b.values[k][j];
        }
    }
}

// Vector product of two matrices(as 1D vectors)
void matrix_dot(matrix a, matrix b, matrix out) {
    if(a.row != out.row || b.row != out.row || a.col != b.col || b.col != out.col || out.col !=  1) {
        printf("Incorrect matrix dimensions [matrix_dot]\n");
        exit(1);
    }
    if(a.values == NULL || b.values == NULL || out.values == NULL) {
        printf("Matrices should not be null [matrix_dot]");
        exit(1);
    }

    for(int i = 0; i < a.row; i++)
        out.values[i][0] = a.values[i][0] * b.values[i][0];
}

// Stores the transpose of matrix in in out
void matrix_transpose(matrix in, matrix out) {
    if(in.row != out.col || in.col != out.row) {
        printf("Incorrect matrix dimensions [transpose]\n");
        printf("[%d %d] [%d %d]\n", in.row, in.col, out.row, out.col);
        exit(1);
    }
    if(in.values == NULL || out.values == NULL) {
        printf("Matrices should not be null [transpose]");
        exit(1);
    }
    if(in.values == out.values) {
        printf("transpose cannot be stored in the same matrix currently");
        exit(1);
    }

    // double temp;

    for(int i = 0; i < in.row; i++) {
        for(int j = 0; j < in.col; j++) {
            // temp = in.values[i][j];
            // out.values[i][j] = in.values[j][i];
            // out.values[j][i] = temp;
            out.values[j][i] = in.values[i][j];
        } 

    }
}

void matrix_sigmoid(matrix in, matrix out) {
    if(in.row != out.row || in.col != out.col) {
        printf("Incorrect matrix dimensions [sigmoid]\n");
        printf("[%d %d] [%d %d]\n", in.row, in.col, out.row, out.col);
        exit(1);
    }
    if(in.values == NULL || out.values == NULL) {
        printf("Matrices shoulb not be null [sigmoid]");
        exit(1);
    }

    for(int i = 0; i < in.row; i++) {
        for(int j = 0; j < in.col; j++) 
            out.values[i][j] = 1.0 /  (1.0 + exp(-in.values[i][j]));
    }
}

void matrix_sigmoid_prime(matrix in, matrix out) {
    if(in.row != out.row || in.col != out.col) {
        printf("Incorrect matrix dimensions [sigmoid_prime]\n");
        printf("[%d %d] [%d %d]\n", in.row, in.col, out.row, out.col);
        exit(1);
    }
    if(in.values == NULL || out.values == NULL) {
        printf("Matrices shoulb not be null [sigmoid_prime]");
        exit(1);
    }

    double epow;

    for(int i = 0; i < in.row; i++) {
        for(int j = 0; j < in.col; j++) {
            epow = exp(-in.values[i][j]);
            out.values[i][j] = epow / pow(1.0 + epow, 2.0);
        }
    }
}

void matrix_random(matrix m, float start, float end) {
    if(m.values == NULL) {
        printf("Matrices shoulb not be null [random_matrix]");
        exit(1);
    }
    if(start >= end) {
        printf("Invalid range in [random_matrix]\n");
        exit(1);
    }

    for(int i = 0; i < m.row; i++) {
        for(int j = 0; j < m.col; j++) {
            m.values[i][j] = start + ((double)rand() / RAND_MAX) * (end - start);
        }
    }
}
