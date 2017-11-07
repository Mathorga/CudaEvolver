#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <curand.h>
#include <curand_kernel.h>
#include "../hpc.h"

#define BLOCK_SIZE 32
#define CHECKS_NUM 15
#define MUTATION_RATE 0.1
#define CROSS_RATE 1

#define cudaCheckError() {                                                                                  \
            cudaError_t e = cudaGetLastError();                                                             \
            if (e != cudaSuccess) {                                                                         \
                printf("Cuda failure %s(%d): %d(%s)\n", __FILE__, __LINE__ - 1, e, cudaGetErrorString(e));  \
                exit(0);                                                                                    \
            }                                                                                               \
        }

// Translates bidimensional indexes to a monodimensional one.
// |i| is the column index.
// |j| is the row index.
// |n| is the number of columns (length of the rows).
#define IDX(i, j, n) ((i) * (n) + (j))

typedef unsigned char cell_t;

enum CellContent {
    EMPTY = 0,
    CHECK = 1,
    PATH = 2
};

struct Point2D {
    int x = -1;
    int y = -1;
    int id = -1;
};

struct Individual {
    Point2D path[CHECKS_NUM];
    float score = 0.0;
};

// Code taken from Rosettacode:
// https://rosettacode.org/wiki/Bitmap/Bresenham%27s_line_algorithm#C.2B.2B
// Implementing Bresenham’s line drawing algorithm.
void drawLine(cell_t *field, int n, int x0, int y0, int x1, int y1) {
    int x = x0;
    int y = y0;

    int dx = abs(x1 - x0), sx = x0 < x1 ? 1 : -1;
    int dy = abs(y1 - y0), sy = y0 < y1 ? 1 : -1;
    int err = (dx > dy ? dx : -dy) / 2;
    int e2;

    for (;;) {
        if (!(x == x0 && y == y0) && !(x == x1 && y == y1)) {
            field[IDX(x, y, n)] = PATH;
        }
        if (x == x1 && y == y1) {
            break;
        }
        e2 = err;
        if (e2 > -dx) {
            err -= dy;
            x += sx;
        }
        if (e2 < dy) {
            err += dx;
            y += sy;
        }
    }
}

void dump(const cell_t *field, const Point2D *path, unsigned int n, unsigned int checksNum, const char *filename) {
    cell_t *fieldCopy = (cell_t *) malloc(n * n * sizeof(cell_t));
    for (unsigned int x = 0; x < n; x++) {
        for (unsigned int y = 0; y < n; y++) {
            fieldCopy[IDX(x, y, n)] = field[IDX(x, y, n)];
        }
    }

    if (path != NULL) {
        for (unsigned int i = 0; i < checksNum; i++) {
            drawLine(fieldCopy, n, path[i].x, path[i].y, path[(i + 1) % checksNum].x, path[(i + 1) % checksNum].y);
        }
    }
    FILE *out = fopen(filename, "w");
    if (out == NULL) {
        printf("Cannot create \"%s\"\n", filename);
        abort();
    }
    fprintf(out, "P6\n");
    fprintf(out, "%d %d\n", n, n);
    fprintf(out, "255\n");
    for (unsigned int x = 0; x < n; x++) {
        for (unsigned int y = 0; y < n; y++) {
            if (field[IDX((x + 1) % n, y, n)] == CHECK ||
                field[IDX(x, (y + 1) % n, n)] == CHECK ||
                field[IDX((x + 1) % n, (y + 1) % n, n)] == CHECK ||
                field[IDX((x - 1 + n) % n, y, n)] == CHECK ||
                field[IDX(x, (y - 1 + n) % n, n)] == CHECK ||
                field[IDX((x - 1 + n) % n, (y - 1 + n) % n, n)] == CHECK ||
                field[IDX((x + 1) %n, (y - 1 + n) % n, n)] == CHECK ||
                field[IDX((x - 1 + n) %n, (y + 1) % n, n)] == CHECK) {
                fprintf(out, "%c%c%c", 255, 30, 30);
            } else if (field[IDX(x, y, n)] == CHECK) {
                fprintf(out, "%c%c%c", 20, 20, 0);
            } else if (fieldCopy[IDX(x, y, n)] == EMPTY) {
                fprintf(out, "%c%c%c", 20, 20, 20);
            } else if (fieldCopy[IDX(x, y, n)] == PATH) {
                fprintf(out, "%c%c%c", 250, 175, 53);
            } else {
                printf("Unknown cell state (%d) of cell %d-%d", fieldCopy[IDX(x, y, n)], x, y);
                abort();
            }
        }
    }
    fclose(out);
}












void initialize(Individual *pop, unsigned int popSize, Point2D *checks, unsigned int checksNum) {
    Point2D *checksCopy = (Point2D *) malloc(checksNum * sizeof(Point2D));
    for (unsigned int i = 0; i < popSize; i++) {
        for (unsigned int j = 0; j < checksNum; j++) {
            checksCopy[j] = checks[j];
        }
        for (unsigned int j = 0; j < checksNum; j++) {
            int index = rand() % (checksNum - j);
            pop[i].path[j] = checksCopy[index];
            for (unsigned int k = index; k < checksNum - 1; k++) {
                checksCopy[k] = checksCopy[k + 1];
            }
        }
    }
}

__device__ void evaluate(Individual *family, unsigned int checksNum) {
    family[threadIdx.x].score = 0;
    for (unsigned int i = 0; i < checksNum; i++) {
        family[threadIdx.x].score +=
        sqrtf(powf(fabsf(family[threadIdx.x].path[(i + 1) % checksNum].x - family[threadIdx.x].path[i].x), 2) +
              powf(fabsf(family[threadIdx.x].path[(i + 1) % checksNum].y - family[threadIdx.x].path[i].y), 2));
    }
    // printf("Score of Individual %d of Family %d: %f\n", threadIdx.x, blockIdx.x, family[threadIdx.x].score);
}

__device__ int select(Individual *family, curandState_t *state) {
    int random = (int) (curand_uniform(state) * blockDim.x);
    return (family[threadIdx.x].score < family[random].score) ? threadIdx.x : random;
}

__device__ void mutate(Individual *family, curandState_t *state) {
    Point2D tmp[CHECKS_NUM];
    for (int i = 0; i < CHECKS_NUM; i++) {
        if (curand_uniform(state) <= MUTATION_RATE) {
            int firstIndex = i;
            int secondIndex = curand_uniform(state) * (CHECKS_NUM - 1);
            for (int j = 0; j < CHECKS_NUM; j++) {
                tmp[j] = family[threadIdx.x].path[j];
            }
            family[threadIdx.x].path[firstIndex] = family[threadIdx.x].path[secondIndex];
            family[threadIdx.x].path[secondIndex] = tmp[firstIndex];
        }
    }
}

__global__ void evolve(Individual *pop, unsigned int genNum, unsigned int checksNum) {
    curandState_t state;
    extern __shared__ Individual family[];
    // Individual *tmpFamily = &family[blockDim.x];

    // Initialize the random number generator.
    curand_init((unsigned long) clock(), blockIdx.x, threadIdx.x, &state);

    // Copy the family to shared memory.
    family[threadIdx.x] = pop[IDX(blockIdx.x, threadIdx.x, blockDim.x)];

    for (unsigned int g = 0; g < genNum; g++) {
        evaluate(family, checksNum);
        __syncthreads();
        // if (g % 10 == 0) {
        //     migrate();
        // }
        select(family, &state);
        __syncthreads();
        // crossover();
        mutate(family, &state);
        __syncthreads();
    }
    pop[IDX(blockIdx.x, threadIdx.x, blockDim.x)] = family[threadIdx.x];
}















int main(int argc, char const *argv[]) {
    Individual *population;
    // float *scores;

    Individual *d_population;
    Individual *d_tmpPop;
    // float *d_scores;

    unsigned int fieldSize = 500;
    unsigned int popSize = 1024;
    unsigned int famNumber = 32;
    unsigned int genNumber = 1000;

    // char fileName[200];
    double startTime = 0.0;
    double endTime = 0.0;
    cell_t *field;
    Point2D *checks;

    if (argc > 4) {
        printf("Usage: %s [fieldSize [popSize [genNumber]]\n", argv[0]);
        return -1;
    }
    if (argc > 1) {
        fieldSize = atoi(argv[1]);
    }
    if (argc > 2) {
        popSize = atoi(argv[3]);
    }
    if (argc > 3) {
        genNumber = atoi(argv[4]);
    }

    // Create a field of checks.
    field = (cell_t *) malloc(fieldSize * fieldSize * sizeof(cell_t));
    checks = (Point2D *) malloc(CHECKS_NUM * sizeof(Point2D));

    for (unsigned int i = 0; i < fieldSize * fieldSize; i++) {
        field[i] = EMPTY;
    }

    srand(time(NULL));
    for (unsigned int i = 0; i < CHECKS_NUM; i++) {
        checks[i].x = (rand() % fieldSize);
        checks[i].y = (rand() % fieldSize);
        checks[i].id = i;
        field[IDX(checks[i].x, checks[i].y, fieldSize)] = true;
    }

    dump(field, NULL, fieldSize, CHECKS_NUM, "field.ppm");

    printf("Field:\n");
    for (unsigned int i = 0; i < CHECKS_NUM; i++) {
        printf("x:%d\ty:%d\n", checks[i].x, checks[i].y);
    }



    const size_t size = popSize * sizeof(Individual);

    dim3 members(popSize / famNumber);
    dim3 families(famNumber);
    size_t sharedMemSize = members.x * sizeof(Individual);

    // Create the host population.
    population = (Individual *) malloc(size);
    // scores = (float *) malloc(popSize);

    // Create the device populations.
    cudaMalloc(&d_population, size);
    cudaMalloc(&d_tmpPop, size);
    // cudaMalloc(&d_scores, popSize * sizeof(float));

    // Initialize the population.
    initialize(population, popSize, checks, CHECKS_NUM);

    // Copy the host population to the device.
    cudaMemcpy(d_population, population, size, cudaMemcpyHostToDevice);



    // ***Execution.***
    printf("Execution:\n");
    startTime = hpc_gettime();

    evolve<<<families, members, sharedMemSize>>>(d_population, genNumber, CHECKS_NUM);
    cudaDeviceSynchronize();

    endTime = hpc_gettime();
    printf("Execution time (s):%f\n\n", endTime - startTime);




    // Copy the device population back to the host.
    cudaMemcpy(population, d_population, size, cudaMemcpyDeviceToHost);
    for (int i = 0; i < popSize; i++) {
        char fileName[255];
        sprintf(fileName, "Individual%d.ppm", i);
        // dump(field, population[0].path, fieldSize, CHECKS_NUM, fileName);
    }





    return 0;
}
