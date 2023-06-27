#ifndef MLP_H
#define MLP_H

#include "matrix.h"

network network_create(int *layers, int count, void (*activation) (matrix, matrix), void (*activation_prime) (matrix, matrix));
matrix network_feed_forward(network net, matrix input);
network network_backpropogate(network net, matrix x, matrix y);

void network_update_mini_batch(network net, dataset train,
    int batch_start, int batch_len, double learning_rate);

void network_stochastic_gradient_descent(network net, dataset train, int epochs,
    int batch_size, double learning_rate, dataset test, double (*evaluate) (network, dataset));

void cost_derivative(matrix output_activation, matrix y, matrix out);

#endif // MLP_H
