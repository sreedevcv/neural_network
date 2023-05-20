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
    for(int i = 0; i <50; i++) {
        // _train.x[i] = malloc(sizeof(matrix));s
        // _train.x[i] = matrix_create(784 , 1);
        // _train.y[i] = matrix_create(10, 1);
        // printf("%d %d\n", _train.x[i].row, _train.x[i].col);

        matrix a = matrix_create(10 , 10);
        // matrix b = matrix_create(10, 1);
        printf("%d %d\n", a.row, a.col);
    }
    // int layers[] = {3, 4, 1};
    // network net = network_create(layers, sizeof(layers) / sizeof(layers[0]));
    // matrix *x_set, * y_set;
    // printf("%ld", sizeof(lodouble));
    // create_polynomial_data(10, 3, &x_set, &y_set);

    // for(int i = 0; i < 10; i++) 
    //     matrix_print(x_set[i]);

    // matrix_print(net.biases[1]);;
    // matrix_print(net.weights[0]);;

    // for(int i = 0; i < 10; i++) {
    //     matrix a = network_feed_forward(net, x_set[i]);
    //     matrix_print(a);
    //     matrix_print(y_set[i]);
    // }

    // matrix a = network_feed_forward(net, x_set[0]);
    // matrix_print(a);
}
