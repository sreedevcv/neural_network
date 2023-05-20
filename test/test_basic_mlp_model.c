#include "../src/matrix.h"
#include "../src/mlp.h"
#include <stdio.h>
#include <stdlib.h>

void create_polynomial_data(int data_size, int features, matrix **x_set, matrix **y_set) {
    matrix *x_data = malloc(data_size * sizeof(matrix));
    matrix *y_data = malloc(data_size * sizeof(matrix));
    double *coefs = malloc(features * sizeof(double));
    double y, x;

    for(int i = 0; i < features; i++)
        coefs[i] = ((double) rand() / RAND_MAX);

    for(int i = 0; i < data_size; i++) {
        x_data[i] = matrix_create(features , 1);
        y_data[i] = matrix_create(1, 1);

        y = 0;
        for(int j = 0; j < features; j++) {
            x = ((double) rand() / RAND_MAX);
            y += (coefs[j] * x) + ((double) rand() / RAND_MAX);
            x_data[i].values[j][0] = x;
        }
        y_data[i].values[0][0] = y / 4;
    }

    *x_set = x_data;
    *y_set = y_data;
}

int main() {

    matrix a = matrix_create(5, 4);
    matrix_random(a, 10, 100);
    matrix_print(a, "");

    matrix_fill(a, 0);
    matrix_print(a, "");
}
