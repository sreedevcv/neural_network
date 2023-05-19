#ifndef MATRIX_H
#define MATRIX_H

typedef double matrix_data_type;

typedef struct _matrix {
    double **values;
    int row, col;
} matrix;

typedef struct _dataset {
    int size;
    matrix *x, *y;
} dataset;

matrix matrix_create(int row, int col);
void matrix_free(matrix m);
matrix matrix_copy(matrix in);
void matrix_print(matrix a, char *msg);

void matrix_add(matrix a, matrix b, matrix c);
void matrix_sub(matrix a, matrix b, matrix c);
void matrix_scalar_multiply(matrix a, matrix_data_type scalar, matrix b);
void matrix_multiply(matrix a, matrix b, matrix c);
void matrix_dot(matrix a, matrix b, matrix out);
void matrix_transpose(matrix in, matrix out);

void matrix_sigmoid(matrix in, matrix out);
void matrix_sigmoid_prime(matrix in, matrix out);
void matrix_random(matrix m, float start, float end);

#endif // MATRIX_H
