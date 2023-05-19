#include "mnist.h"
#include "../src/matrix.h"
#include "../src/mlp.h"


void print_mnist_image(int index) {
	int i;
	for (i=0; i<784; i++) {
		printf("%1.1f ", test_image[index][i]);
		if ((i+1) % 28 == 0) putchar('\n');
	}

	printf("label: %d\n", test_label[index]);
}

double mnist_evaluate(network net, dataset test) {
    int actual_max = 0, expected_max = 0;
    int accuracy = 0;

    for(int i = 0; i < test.size; i++) {
        matrix actual_y = network_feed_forward(net, test.x[i]);

        for(int j = 0; j < actual_y.row; j++) {
            if(actual_y.values[j][0] > actual_y.values[actual_max][0])
                actual_max = j;

            if(test.y[i].values[j][0] > test.y[i].values[expected_max][0])
                expected_max = j;
        }

        if(actual_max == expected_max)
            accuracy++;
    }

    // printf("acc: %f\n", (double) accuracy / test.size);

    return (double) accuracy / test.size;
}

void convert_to_matrix(int train_size, int test_size, dataset *test, dataset *train) {
    dataset _train, _test;
    _train.size = train_size;
    _test.size = test_size;
    _train.x = malloc(train_size * sizeof(matrix));
    _train.y = malloc(train_size * sizeof(matrix));
    _test.x = malloc(test_size * sizeof(matrix));
    _test.y = malloc(test_size * sizeof(matrix));

    for(int i = 0; i < train_size; i++) {
        
        matrix a = matrix_create(784 , 1);
        matrix b = matrix_create(10, 1);
        // _train.x[i] = matrix_create(784 , 1);
        // _train.y[i] = matrix_create(10, 1);
        // matrix_print(_train.x[i], "tx");
        // matrix_print(_train.y[i], "ty");

        matrix_print(a, "tx");
        matrix_print(b, "ty");

        for(int j = 0; j < 784; j++)
            _train.x[i].values[j][0] = train_image[i][j];

        for(int j = 0; j < 10; j++) {
            if(train_label[i] == j)
                _train.y[i].values[j][0] = (matrix_data_type) 1;
            else
                _train.y[i].values[j][0] = (matrix_data_type) 0;
        }
    }

    for(int i = 0; i < test_size; i++) {
        _test.x[i] = matrix_create(784 , 1);
        _test.y[i] = matrix_create(10, 1);

        for(int j = 0; j < 784; j++)
            _test.x[i].values[j][0] = test_image[i][j];

        _test.y[i].values[test_label[i]][0] = 5.0;
        // _test.y[i].values[test_label[i]][0] = (matrix_data_type) 5.0L;

        // printf("label: %d", test_label[i]);
        // matrix_print(_test.y[i], "test");
    }


    *test = _test;
    *train = _train;
    // for(int i = 0; i < test_size; i++)
    //     printf("%p\n", _test.y[i].values);
}

struct temp
{
    int a, b;
    char c;
};


int main() {
	load_mnist();

    int layers[] = {784, 30, 10};
    // network net = network_create(layers, 3);
    // int train_size = 5000, test_size = 5000;
    int train_size = 2, test_size = 5;
    dataset test, train;
    // convert_to_matrix(train_size, test_size, &test, &train);

    matrix *a = malloc(sizeof(matrix));
    a[0] = matrix_create(3, 3);
    matrix_print(a[0], "");

    // for(int i = 0; i < 5; i++) {
    //     // matrix_print(train.x[i]);
    //     // print_mnist_image()
    // }

    // for(int i = 0; i < 2; i++)
    //     matrix_print(net.biases[i], "b");

    // for(int i = 0; i < 2; i++)
    //     matrix_print(net.weights[i], "w");

    // network_stochastic_gradient_descent(net, train, 1, 50, 3.0, test, mnist_evaluate);
    // network_stochastic_gradient_descent(net, train, 1, 1, 3.0, test, mnist_evaluate);

    // matrix_print(network_feed_forward(net, test.x[100]), "test");
    // matrix_print(test.y[2], "result");
    // printf("Label :%d\n", test_label[1]);

    // printf("%Lf",(matrix_data_type) 1.0);

    // print_mnist_image(1800);
    // matrix x = network_feed_forward(net, test.x[1800]);
    // matrix_print(x);

    // for(int i = 0; i < 5; i++)
    //     matrix_print(test.y[i], "");
	return 0;
}

/*
int train_size = 50000, test_size = 5000;
network_stochastic_gradient_descent(net, train, 40, 50, 3.0, test, mnist_evaluate);
Epoch 0: 79.380000
Epoch 1: 81.920000
Epoch 2: 82.760000
Epoch 3: 83.300000
Epoch 4: 83.660000
Epoch 5: 83.900000
Epoch 6: 84.280000
Epoch 7: 84.460000
Epoch 8: 84.560000
Epoch 9: 84.580000
Epoch 10: 84.680000
Epoch 11: 84.660000
Epoch 12: 84.620000
Epoch 13: 84.560000
Epoch 14: 84.640000
Epoch 15: 84.800000
Epoch 16: 84.880000
Epoch 17: 84.960000
Epoch 18: 85.080000
Epoch 19: 85.060000
Epoch 20: 85.080000
Epoch 21: 85.080000
Epoch 22: 85.060000
Epoch 23: 85.140000
Epoch 24: 85.080000
Epoch 25: 85.080000
Epoch 26: 85.040000
Epoch 27: 85.000000
Epoch 28: 85.020000
Epoch 29: 85.020000
Epoch 30: 84.940000
Epoch 31: 84.940000
Epoch 32: 84.920000
Epoch 33: 84.920000
Epoch 34: 84.980000
Epoch 35: 85.060000
Epoch 36: 85.060000
Epoch 37: 85.000000
Epoch 38: 85.000000
Epoch 39: 85.000000

py: 9426 / 10000
*/