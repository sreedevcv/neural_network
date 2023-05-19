#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <time.h>
#include "matrix.h"

typedef struct _network {
    int layer_count, *layers;
    matrix *biases, *weights;
} network;

// Create a mlp neural netwrok
network network_create(int *layers, int count) {
    network net;
    net.layer_count = count;
    net.layers = layers;
    net.biases = malloc((count - 1) * sizeof(matrix));
    net.weights = malloc((count - 1) * sizeof(matrix));

    for(int i = 0; i < count - 1; i++) {
        net.biases[i] = matrix_create(layers[i + 1], 1);
        net.weights[i] = matrix_create(layers[i + 1], layers[i]);

        matrix_random(net.biases[i], -1, 1);
        matrix_random(net.weights[i], -1, 1);
    }

    return net;
}

// Feeds the given input through the layer
matrix network_feed_forward(network net, matrix input) {
    matrix out;
    matrix in = matrix_copy(input);

    for(int i = 0; i < net.layer_count - 1; i++) {
        out = matrix_create(net.weights[i].row, in.col);
        matrix_multiply(net.weights[i], in, out);
        matrix_add(out, net.biases[i], out);
        matrix_sigmoid(out, out);

        matrix_free(in);
        in = out;
    }

    return out;
}

void cost_derivative(matrix output_activation, matrix y, matrix out) {
    matrix_sub(output_activation, y, out);
}

network network_backpropogate(network net, matrix x, matrix y) {
    matrix *nabla_b = malloc((net.layer_count - 1) * sizeof(matrix));
    matrix *nabla_w = malloc((net.layer_count - 1) * sizeof(matrix));
    matrix *activations = malloc((net.layer_count) * sizeof(matrix));
    matrix *zs = malloc((net.layer_count - 1) * sizeof(matrix));

    matrix delta = matrix_create(y.row, y.col);
    matrix temp = matrix_create(y.row, y.col);
    matrix activation = matrix_copy(x);


    for (int i = 0; i < net.layer_count - 1; i++) {
        nabla_b[i] = matrix_create(net.biases[i].row, net.biases[i].col);
        nabla_w[i] = matrix_create(net.weights[i].row, net.weights[i].col);
    }

    activations[0] = activation;

    // Finds the activation and z values of each layer
    // z = weight * activation + b
    // activation = sigmoid(z)
    for (int i = 0; i < net.layer_count - 1; i++) {
        // printf("zs: %d\n", i);
        zs[i] = matrix_create(net.weights[i].row, activation.col);

        // matrix_print(net.weights[i], "weight");
        // matrix_print(activation, "activation");
        matrix_multiply(net.weights[i], activation, zs[i]);
        // matrix_print(zs[i], "zs[i] mult");
        

        matrix_add(zs[i], net.biases[i], zs[i]);
        // matrix_print(zs[i], "zs[i] add");

        activation = matrix_create(zs[i].row, zs[i].col);
        matrix_sigmoid(zs[i], activation);
        // matrix_print(activation, "activation sig");
        activations[i + 1] = activation;
        // matrix_print(activation, "out");


    }

    // Calculates delta
    // delta = cost_derivative(final_activation, y) * sigmoid_pirme(final_z)
    cost_derivative(activations[net.layer_count - 1], y, delta);
    matrix_sigmoid_prime(zs[net.layer_count - 2], temp);
    matrix_dot(delta, temp, delta);
    matrix_free(temp);

    // TODO::current valus of nabla_b is replaced and not deallocated
    nabla_b[net.layer_count - 2] = delta;

    // last_nabla_w = delta * transpose(penultimate_activation)
    temp = matrix_create(activations[net.layer_count - 2].col, activations[net.layer_count - 2].row);
    matrix_transpose(activations[net.layer_count - 2], temp);
    matrix_multiply(delta, temp, nabla_w[net.layer_count - 2]);
    matrix_free(temp);
    matrix_free(activations[net.layer_count - 2]);
    // matrix_free(delta);

    // printf("hai\n");
    //     for(int i = 0; i < 2; i++)
    //     matrix_print(nabla_b[i], "b");

    // for(int i = 0; i < 2; i++)
    //     matrix_print(nabla_w[i], "w");

    for(int i = net.layer_count - 3; i >= 0; i--) {
        // delta = matrix_create(zs[i].row, zs[i].col);
        temp = matrix_create(net.weights[i + 1].col, net.weights[i + 1].row);
        matrix_sigmoid_prime(zs[i], zs[i]);
        matrix_transpose(net.weights[i + 1], temp);
        matrix_multiply(temp, delta, nabla_b[i]);

        matrix_dot(nabla_b[i], zs[i], nabla_b[i]);
        matrix_free(temp);

        delta = nabla_b[i];
        temp = matrix_create(activations[i].col, activations[i].row);
        matrix_transpose(activations[i], temp);
        matrix_multiply(delta, temp, nabla_w[i]);

        // printf("hai\n");

        matrix_free(temp);
        matrix_free(zs[i]);
        matrix_free(activations[i]);
    }

    network n;
    n.biases = nabla_b;
    n.weights = nabla_w;

    // for(int i = 0; i < 2; i++)
    //     matrix_print(nabla_b[i], "b");

    // for(int i = 0; i < 2; i++)
    //     matrix_print(nabla_w[i], "w");

    return n;
}

void network_update_mini_batch(network net, dataset train,
    int batch_start, int batch_len, double learning_rate) {

    matrix *nabla_b = malloc((net.layer_count - 1) * sizeof(matrix));
    matrix *nabla_w = malloc((net.layer_count - 1) * sizeof(matrix));
    network delta;

    for (int i = 0; i < net.layer_count - 1; i++) {
        nabla_b[i] = matrix_create(net.biases[i].row, net.biases[i].col);
        nabla_w[i] = matrix_create(net.weights[i].row, net.weights[i].col);
    }

    for(int i = batch_start; i < batch_start + batch_len; i++) {
        delta = network_backpropogate(net, train.x[i], train.y[i]);
        // printf("back\n");

        for (int j = 0; j < net.layer_count - 1; j++) {
            // printf("||%d||\n", j);
            matrix_add(nabla_b[j], delta.biases[j], nabla_b[j]);
            matrix_add(nabla_w[j], delta.weights[j], nabla_w[j]);
            matrix_free(delta.biases[j]);
            matrix_free(delta.weights[j]);

        }

        free(delta.biases);
        free(delta.weights);
    }

    for (int i = 0; i < net.layer_count - 1; i++) {
        matrix_scalar_multiply(nabla_b[i], learning_rate / (double) batch_len, nabla_b[i]);
        matrix_sub(net.biases[i], nabla_b[i], net.biases[i]);

        matrix_scalar_multiply(nabla_w[i], learning_rate / (double) batch_len, nabla_w[i]);
        matrix_sub(net.weights[i], nabla_w[i], net.weights[i]);

        matrix_free(nabla_b[i]);
        matrix_free(nabla_w[i]);

        // matrix_print(net.biases[i]);
    }

    free(nabla_b);
    free(nabla_w);
}

void network_stochastic_gradient_descent(network net, dataset train, int epochs, 
    int batch_size, double learning_rate, dataset test, double (*evaluate) (network, dataset)) {
    
    bool test_set_avaliable = false;
    double accuracy;

    if(test.x != NULL && test.y != NULL && evaluate != NULL) {
        test_set_avaliable = true;
    }

    for(int i = 0; i < epochs; i++) {
        // printf("ts %d, bs %d\n", train.size, batch_size);
        clock_t tic = clock();
        for(int j = 0; j < train.size; j += batch_size) {
            network_update_mini_batch(net, train, j, batch_size, learning_rate);
        }
        clock_t toc = clock();

        if(test_set_avaliable) {
            accuracy = evaluate(net, test);
            printf("Epoch %d: %f, time: %lf\n", i, accuracy * 100, (toc - tic) / CLOCKS_PER_SEC);
        }
        else
            printf("Epoch %d completed\n", i);
    }
}
