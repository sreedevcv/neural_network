CC=gcc
CFLAGS= -g -Wall -Wextra

mnist: src/matrix.c src/mlp.c test/mnist_model.c
	$(CC) $(CFLAGS) -o mnist test/mnist_model.c src/matrix.c src/mlp.c -lm