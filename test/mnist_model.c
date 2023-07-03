#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <unistd.h>
#include <sys/wait.h>
#include "mnist.h"
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

    return (double) accuracy / test.size;
}


// Loads the dataset test and train with mnist images
void convert_to_matrix(int train_size, int test_size, dataset *test, dataset *train) {
    dataset _train, _test;
    _train.size = train_size;
    _test.size = test_size;
    _train.x = (matrix*) malloc(train_size * sizeof(matrix));
    _train.y = (matrix*) malloc(train_size * sizeof(matrix));
    _test.x = (matrix*) malloc(test_size * sizeof(matrix));
    _test.y = (matrix*) malloc(test_size * sizeof(matrix));


    for(int i = 0; i < train_size; i++) {
        _train.x[i] = matrix_create(784 , 1);
        _train.y[i] = matrix_create(10, 1);

        for(int j = 0; j < 784; j++)
            _train.x[i].values[j][0] = train_image[i][j];

        for(int j = 0; j < 10; j++) {
            if(train_label[i] == j)
                _train.y[i].values[j][0] = (double) 1;
            else
                _train.y[i].values[j][0] = (double) 0;
        }
    }

    for(int i = 0; i < test_size; i++) {
        _test.x[i] = matrix_create(784 , 1);
        _test.y[i] = matrix_create(10, 1);

        for(int j = 0; j < 784; j++)
            _test.x[i].values[j][0] = test_image[i][j];

        _test.y[i].values[test_label[i]][0] = 1.0;
    }

    *test = _test;
    *train = _train;
}

/* Checks if the mnist dataset exixts If not, downloads them using curl 
 * and unzips them using gzip
*/
void download_mnist_dataset() {
    int curl_pid, gzip_pid, status;
    char *base_url = "http://yann.lecun.com/exdb/mnist/";
    char *fileNames[] = {"train-images-idx3-ubyte", "train-labels-idx1-ubyte",
                        "t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte"};
    char name[100];
    char url[100];


    for(int i = 0; i < 4; i++) {
        sprintf(name, "./archive/%s", fileNames[i]);

        if((access(name, F_OK)) != 0) {
            printf("%s does not exist\n", name);
            sprintf(url, "%s%s.gz", base_url, fileNames[i]);
            strcat(name, ".gz");

            curl_pid = fork();
            if (curl_pid == 0) {
                printf("Downloading %s\n", url);
                execl("/usr/bin/curl", "curl", url, "--output", name, NULL);
            }
            else 
                wait(&status);

            gzip_pid = fork();

            if(gzip_pid == 0) {
                printf("Unzipping %s\n", name);
                execl("/usr/bin/gzip", "gzip", "-d", name, NULL);
            }
            else 
                wait(&status);

        }
    }
}

double sigmoid(double num) {
    return 1.0 /  (1.0 + exp(-num));
}

double sigmoid_prime(double num) {
    double epow = exp(-num);
    return epow / pow(1.0 + epow, 2.0);
}

int main() {
    download_mnist_dataset();
	load_mnist();
    srand(time(0));
    int layers[] = {784, 30, 10};
    network net = network_create(layers, 3, sigmoid, sigmoid_prime);
    int train_size = 60000, test_size = 10000;
    dataset test, train;
    convert_to_matrix(train_size, test_size, &test, &train);
    printf("Loaded mnist dataset\n");


    network_stochastic_gradient_descent(net, train, 40, 50, 1.0, test, mnist_evaluate);
    // network_stochastic_gradient_descent(net, train, 4, 30, 3.0, test, mnist_evaluate);

    // matrix_print(network_feed_forward(net, test.x[100]), "test");
    // matrix_print(test.y[2], "result");
    // printf("Label :%d\n", test_label[1]);
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

/*
Epoch 0: 86.740000, time: 21.326435
Epoch 1: 89.800000, time: 18.566099
Epoch 2: 90.780000, time: 18.492725
Epoch 3: 91.220000, time: 18.527351
Epoch 4: 91.680000, time: 18.506668
Epoch 5: 92.040000, time: 18.545433
Epoch 6: 92.260000, time: 18.440934
Epoch 7: 92.500000, time: 18.488570
Epoch 8: 92.640000, time: 18.596878
Epoch 9: 92.800000, time: 18.690502
Epoch 10: 92.980000, time: 18.543569
Epoch 11: 93.140000, time: 18.697627
Epoch 12: 93.280000, time: 18.542397
Epoch 13: 93.360000, time: 18.547516
Epoch 14: 93.400000, time: 18.469518
Epoch 15: 93.440000, time: 18.550860

After removing malloc and free from backprop
Epoch 0: 86.500000, time: 16.057375
Epoch 1: 89.620000, time: 13.142268
Epoch 2: 90.820000, time: 13.139444
Epoch 3: 91.520000, time: 13.141751
Epoch 4: 91.780000, time: 13.138698
Epoch 5: 92.160000, time: 13.143167
Epoch 6: 92.480000, time: 13.141336
Epoch 7: 92.780000, time: 13.143494
Epoch 8: 92.820000, time: 13.135197
Epoch 9: 93.080000, time: 13.139074



Epoch 0: 86.500000, time: 16.048770
Epoch 1: 89.620000, time: 13.140971
Epoch 2: 90.820000, time: 13.137055
Epoch 3: 91.520000, time: 13.130475
Epoch 4: 91.780000, time: 13.097974
Epoch 5: 92.160000, time: 13.069101
Epoch 6: 92.480000, time: 13.181079
Epoch 7: 92.780000, time: 13.197024

*/

/*
network_stochastic_gradient_descent(net, train, 40, 50, 1.0, test, mnist_evaluate);
Epoch 0, [84.750000], time: 2.927666 sec
Epoch 1, [89.270000], time: 2.928950 sec
Epoch 2, [90.840000], time: 2.935688 sec
Epoch 3, [91.850000], time: 2.935794 sec
Epoch 4, [92.380000], time: 2.936768 sec
Epoch 5, [92.830000], time: 2.937061 sec
Epoch 6, [93.140000], time: 2.938567 sec
Epoch 7, [93.290000], time: 2.939554 sec
Epoch 8, [93.450000], time: 2.980682 sec
Epoch 9, [93.690000], time: 3.080716 sec
Epoch 10, [93.800000], time: 3.043722 sec
Epoch 11, [93.900000], time: 2.972642 sec
Epoch 12, [94.030000], time: 2.933663 sec
Epoch 13, [94.260000], time: 2.932719 sec
Epoch 14, [94.330000], time: 2.942734 sec
Epoch 15, [94.420000], time: 2.944672 sec
Epoch 16, [94.510000], time: 2.962797 sec
Epoch 17, [94.550000], time: 2.928447 sec
Epoch 18, [94.630000], time: 3.009058 sec
Epoch 19, [94.700000], time: 2.933164 sec
Epoch 20, [94.760000], time: 3.052855 sec
Epoch 21, [94.770000], time: 3.002120 sec
Epoch 22, [94.800000], time: 3.164965 sec
Epoch 23, [94.850000], time: 2.928945 sec
Epoch 24, [94.840000], time: 2.928599 sec
Epoch 25, [94.860000], time: 2.997378 sec
Epoch 26, [94.910000], time: 3.245240 sec
Epoch 27, [94.920000], time: 3.343054 sec
Epoch 28, [94.950000], time: 3.057187 sec
Epoch 29, [95.010000], time: 3.187030 sec
Epoch 30, [95.020000], time: 3.325045 sec
Epoch 31, [95.020000], time: 2.965155 sec
Epoch 32, [95.030000], time: 2.928220 sec
Epoch 33, [95.060000], time: 2.965973 sec
Epoch 34, [95.120000], time: 2.992003 sec
Epoch 35, [95.140000], time: 3.554348 sec
Epoch 36, [95.150000], time: 3.197144 sec
Epoch 37, [95.190000], time: 3.034788 sec
Epoch 38, [95.180000], time: 2.963449 sec
Epoch 39, [95.170000], time: 3.015731 sec
*/
