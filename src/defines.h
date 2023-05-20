#ifndef DEFINES_H
#define DEFINES_H

typedef struct _matrix {
    int row, col;
    double **values;
} matrix;

typedef struct _dataset {
    int size;
    matrix *x, *y;
} dataset;

typedef struct _network {
    int layer_count, *layers;
    matrix *biases, *weights;
    void (*activation) (matrix in, matrix out);
    void (*activation_prime) (matrix in, matrix out);
} network;

#endif
