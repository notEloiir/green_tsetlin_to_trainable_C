#include <stdint.h>

#include "tsetlin_machine.h"
#include "sparse_tsetlin_machine.h"
#include "stateless_tsetlin_machine.h"


void load_mnist_data(uint8_t *x_data, int32_t *y_data);

void evaluate_models(
    struct TsetlinMachine *tm,
    struct SparseTsetlinMachine *stm,
    struct StatelessTsetlinMachine *sltm,
    uint8_t *x_data,
    int32_t *y_data,
    uint32_t rows
);

void train_models(
    struct TsetlinMachine *tm,
    struct SparseTsetlinMachine *stm,
    uint8_t *x_data,
    int32_t *y_data,
    uint32_t rows
);
