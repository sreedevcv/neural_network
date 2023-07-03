#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <time.h>
#include "matrix.h"

matrix *TRNPS_ACTIVATION;   /* Transpose of the activations of each layer */
matrix *TRNPS_WEIGHTS;      /* Transpose of weights of each layer */
matrix *ACTIVATIONS;        /* Activation of each layer */
matrix *NABLA_B;
matrix *NABLA_W;
matrix *N_B;
matrix *N_W;
matrix *ZS;

/* Create a mlp neural netwrok */
network network_create(int *layers, int count, double (*activation) (double), double (*activation_prime) (double)) {
    network net;
    net.layer_count = count;
    net.layers = layers;
    net.biases = malloc((count - 1) * sizeof(matrix));
    net.weights = malloc((count - 1) * sizeof(matrix));
    net.activation = activation;
    net.activation_prime = activation_prime;

    for(int i = 0; i < count - 1; i++) {
        net.biases[i] = matrix_create(layers[i + 1], 1);
        net.weights[i] = matrix_create(layers[i + 1], layers[i]);

        matrix_random(net.biases[i], -1, 1);
        matrix_random(net.weights[i], -1, 1);
    }

    return net;
}

/* Feeds the given input through the layer */
matrix network_feed_forward(network net, matrix input) {
    matrix out;
    matrix_copy(input, ACTIVATIONS[0]);

    for(int i = 0; i < net.layer_count - 1; i++) {
        out = ACTIVATIONS[i + 1];
        matrix_multiply(net.weights[i], ACTIVATIONS[i], out);
        matrix_add(out, net.biases[i], out);
        matrix_apply(net.activation, out, out);
    }

    return out;
}

/* Derivative of cost function (diffrence squared) */
void cost_derivative(matrix output_activation, matrix y, matrix out) {
    matrix_sub(output_activation, y, out);
}

/* Calculates NABLA_B and NABLA_W */
void network_backpropogate(network net, matrix x, matrix y) {
    matrix temp = matrix_create(y.row, y.col), delta;
    matrix_copy(x, ACTIVATIONS[0]);

    /* Finds the activation and z values of each layer
    * z = weight * activation + b
    * activation = sigmoid(z)
    */
    for (int i = 0; i < net.layer_count - 1; i++) {
        matrix_multiply(net.weights[i], ACTIVATIONS[i], ZS[i]);
        matrix_add(ZS[i], net.biases[i], ZS[i]);
        matrix_apply(net.activation, ZS[i], ACTIVATIONS[i + 1]);
    }

    /* Calculates delta
    * delta = cost_derivative(final_activation, y) * sigmoid_pirme(final_z)
    */
    cost_derivative(ACTIVATIONS[net.layer_count - 1], y, NABLA_B[net.layer_count - 2]);
    matrix_apply(net.activation_prime, ZS[net.layer_count - 2], temp);
    matrix_dot(NABLA_B[net.layer_count - 2], temp, NABLA_B[net.layer_count - 2]);
    matrix_free(temp);

    delta = NABLA_B[net.layer_count - 2];
    /* last_nabla_w = delta * transpose(penultimate_activation) */
    temp = TRNPS_ACTIVATION[net.layer_count - 2];
    matrix_transpose(ACTIVATIONS[net.layer_count - 2], temp);
    matrix_multiply(delta, temp, NABLA_W[net.layer_count - 2]);

    for(int i = net.layer_count - 3; i >= 0; i--) {
        temp = TRNPS_WEIGHTS[i + 1];
        matrix_apply(net.activation_prime, ZS[i], ZS[i]);
        matrix_transpose(net.weights[i + 1], temp);
        matrix_multiply(temp, delta, NABLA_B[i]);
        matrix_dot(NABLA_B[i], ZS[i], NABLA_B[i]);

        delta = NABLA_B[i];
        temp = TRNPS_ACTIVATION[i];
        matrix_transpose(ACTIVATIONS[i], temp);
        matrix_multiply(delta, temp, NABLA_W[i]);
    }
}

/* Runs backpropogation algorithm on a of the entire training set */
void network_update_mini_batch(network net, dataset train,
    int batch_start, int batch_len, double learning_rate) {


    network_backpropogate(net, train.x[batch_start], train.y[batch_start]);
    for (int j = 0; j < net.layer_count - 1; j++) {
        matrix_copy(NABLA_B[j], N_B[j]);
        matrix_copy(NABLA_W[j], N_W[j]);
    }

    for(int i = batch_start + 1; i < batch_start + batch_len; i++) {
        network_backpropogate(net, train.x[i], train.y[i]);

        for (int j = 0; j < net.layer_count - 1; j++) {
            matrix_add(N_B[j], NABLA_B[j], N_B[j]);
            matrix_add(N_W[j], NABLA_W[j], N_W[j]);
        }
    }

    for (int i = 0; i < net.layer_count - 1; i++) {
        matrix_scalar_multiply(N_B[i], learning_rate / (double) batch_len, N_B[i]);
        matrix_sub(net.biases[i], N_B[i], net.biases[i]);

        matrix_scalar_multiply(N_W[i], learning_rate / (double) batch_len, N_W[i]);
        matrix_sub(net.weights[i], N_W[i], net.weights[i]);
    }
}


/* Trains a network using the stochastic gradient descent method */
void network_stochastic_gradient_descent(network net, dataset train, int epochs,
    int batch_size, double learning_rate, dataset test, double (*evaluate) (network, dataset)) {

    bool test_set_avaliable = false;
    int b_size;
    double accuracy;

    if(test.x != NULL && test.y != NULL && evaluate != NULL) {
        test_set_avaliable = true;
    }

    TRNPS_ACTIVATION = malloc((net.layer_count) * sizeof(matrix));
    TRNPS_WEIGHTS = malloc((net.layer_count - 1) * sizeof(matrix));
    ACTIVATIONS = malloc((net.layer_count) * sizeof(matrix));
    NABLA_B = malloc((net.layer_count - 1) * sizeof(matrix));
    NABLA_W = malloc((net.layer_count - 1) * sizeof(matrix));
    N_B = malloc((net.layer_count - 1) * sizeof(matrix));
    N_W = malloc((net.layer_count - 1) * sizeof(matrix));
    ZS = malloc((net.layer_count - 1) * sizeof(matrix));

    TRNPS_ACTIVATION[0] = matrix_create(train.x[0].col, train.x[0].row);
    ACTIVATIONS[0] = matrix_create(train.x[0].row, train.x[0].col);

    for (int i = 0; i < net.layer_count - 1; i++) {
        TRNPS_ACTIVATION[i + 1] = matrix_create(1, net.layers[i + 1]);
        ACTIVATIONS[i + 1] = matrix_create(net.layers[i + 1], 1);
        TRNPS_WEIGHTS[i] = matrix_create(net.weights[i].col, net.weights[i].row);
        NABLA_B[i] = matrix_create(net.biases[i].row, net.biases[i].col);
        NABLA_W[i] = matrix_create(net.weights[i].row, net.weights[i].col);
        N_B[i] = matrix_create(net.biases[i].row, net.biases[i].col);
        N_W[i] = matrix_create(net.weights[i].row, net.weights[i].col);
        ZS[i] = matrix_create(net.weights[i].row, ACTIVATIONS[i].col);
    }

    for(int i = 0; i < epochs; i++) {
        clock_t tic = clock();

        for(int j = 0; j < train.size; j += batch_size) {
            b_size = j + batch_size < train.size ? batch_size : train.size - j;
            network_update_mini_batch(net, train, j, b_size, learning_rate);
        }

        clock_t toc = clock();

        if(test_set_avaliable) {
            accuracy = evaluate(net, test);
            printf("Epoch %d, [%f], time: %lf sec\n", i, accuracy * 100, (double) (toc - tic) / CLOCKS_PER_SEC);
        }
        else
            printf("Epoch %d completed\n", i);
    }

    matrix_free(TRNPS_ACTIVATION[0]);
    matrix_free(ACTIVATIONS[0]);
    for (int i = 0; i < net.layer_count - 1; i++) {
        matrix_free(TRNPS_ACTIVATION[i + 1]);
        matrix_free(ACTIVATIONS[i + 1]);
        matrix_free(TRNPS_WEIGHTS[i]);
        matrix_free(NABLA_B[i]);
        matrix_free(NABLA_W[i]);
        matrix_free(N_B[i]);
        matrix_free(N_W[i]);
        matrix_free(ZS[i]);
    }

    free(TRNPS_ACTIVATION);
    free(TRNPS_WEIGHTS);
    free(ACTIVATIONS);
    free(NABLA_B);
    free(NABLA_W);
    free(N_B);
    free(N_W);
    free(ZS);
}
