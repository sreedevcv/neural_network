#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "../src/matrix.h"
#include "../src/mlp.h"

bool test_addition() {
    int row = 3, col = 4, count = 0;
    matrix a = matrix_create(row, col);
    matrix b = matrix_create(row, col);
    matrix c = matrix_create(row, col);

    for(int i = 0; i < row; i++) {
        for(int j = 0; j < col; j++) {
            a.values[i][j] = ++count;
            b.values[i][j] = row * col - count;
        }
    }

    matrix_add(a, b, c);

    for(int i = 0; i < row; i++) {
        for(int j = 0; j < col; j++) {
            if(c.values[i][j] != row * col)
                return false;
        }
    }

    matrix_print(a, "");
    matrix_print(b, "");
    matrix_print(c, "");

    return true;
}


void test_create_network() {
    int layers[] = {3, 4, 4, 2};
    network n = network_create(layers, 4);

    for(int i = 0; i < 3; i++)
        matrix_print(n.biases[i], "");

    for(int i = 0; i < 3; i++)
        matrix_print(n.weights[i], "");
}

int main()
{
    // srand(time(0));
    // test_create_network();
    // printf("%d", test_addition());
    // matrix a = matrix_create(5, 6);
    // matrix_print(a);
    // matrix_random(a, -10, 10);
    // matrix_print(a);
    // matrix b = a;
    // matrix_print(b);

    // matrix a = matrix_create(3, 4);
    // matrix_random(a, 0, 5);
    // matrix_print(a);
    // matrix_sigmoid(a,a );
    // matrix_print(a);

    printf("%f\n", 1.0 /  (1.0 + exp(-10.669294)));

    return 0;
}

    // printf("%ld\n", sizeof(int));
    // printf("%ld\n", sizeof(float));
    // printf("%ld\n", sizeof(double));
    // printf("%ld\n", sizeof(long double));
    // printf("%ld",  sizeof(__float128));

    // gcc -Wall -Wextra -g test/test_matrix.c src/matrix.c src/mlp.c -lm  