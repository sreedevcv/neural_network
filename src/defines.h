#ifndef DEFINES_H
#define DEFINES_H


typedef struct _matrix {
    int row, col;       /* Rows and columns of the matrix */
    double **values;    /* Actual values of the matrix */
} matrix;

typedef struct _dataset {
    int size;           /* Size of the dataset*/
    matrix *x, *y;      /* Input and output vectors for the dataset */
} dataset;

typedef struct _network {
    int layer_count, *layers;                           /* Number of nuerons in eac layer */
    matrix *biases, *weights;                           /* Biases and Weights of the network */
    void (*activation) (matrix in, matrix out);         /* Activation function to be used */
    void (*activation_prime) (matrix in, matrix out);   /* Derivative of the activation function */
} network;

#endif
