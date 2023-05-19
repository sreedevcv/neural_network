
set -xe
# gcc -Wall -Wextra -g test/mnist_model.c src/matrix.c src/mlp.c -o mt -lm
gcc -Ofast  test/mnist_model.c src/matrix.c src/mlp.c -o mt -lm