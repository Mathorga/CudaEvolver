#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <ga/ga.h>
#include "../hpc.h"
#include "PathGenome.h"
#include "try.h"

// Translates bidimensional indexes to a monodimensional one.
// |i| is the column index.
// |j| is the row index.
// |n| is the number of columns (length of the rows).
#define IDX(i, j, n) ((i) * (n) + (j))

typedef unsigned char cell_t;

enum {
    EMPTY = 0,
    CHECK = 1,
    PATH = 2
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

    for(;;){
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

void dump(const cell_t *field, const PathGenome::_2DDot *path, unsigned int n, unsigned int checksNum, const char *filename) {

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


float serialEvaluator(GAGenome &g) {
    PathGenome &genome = (PathGenome &) g;
    float distance = 0.0;
    float score = 0.0;
    for (unsigned int i = 0; i < genome.getChecksNum(); i++) {
        float dx = (float) genome.gene((i + 1) % genome.getChecksNum()).x - (float) genome.gene(i).x;
        float dy = (float) genome.gene((i + 1) % genome.getChecksNum()).y - (float) genome.gene(i).y;
        distance = sqrtf(powf(dx, 2) + powf(dy, 2));
        // printf("distance %d:%f\n\n", i, distance);
        genome.setDistance(i, distance);
    }
    for (unsigned int i = 0; i < genome.getChecksNum(); i++) {
        // printf("distance %d on host:%f\n\n", i, genome->getDistances()[i]);
        score += genome.getDistances()[i];
    }
    // printf("score:%f\n", score);
    return score;
}


// void populationEvaluator(GAPopulation &pop) {
//     // Copy individuals to device
//     evaluate<<<popSize, checksNumber>>>(pop.individuals());
//     for (int i = 0; i < pop.size(); i++) {
//         pop.individual(i).evaluate(gaTrue);
//     }
// }

int main(int argc, char const *argv[]) {
    hi<<<1, 10>>>();
    cudaDeviceSynchronize();

    unsigned int fieldSize = 500;
    unsigned int checksNumber = 30;
    unsigned int popSize = 100;
    unsigned int genNumber = 1000;
    float mutRate = 0.1;
    float crossRate = 1;

    char fileName[200];
    double startTime = 0.0;
    double endTime = 0.0;
    cell_t *field;
    PathGenome::_2DDot *checks;

    if (argc > 7) {
        printf("Usage: %s [fieldSize [checksNumber [popSize [genNumber [mutRate [crossRate]]]]]]\n", argv[0]);
        return -1;
    }

    if (argc > 1) {
        fieldSize = atoi(argv[1]);
    }

    if (argc > 2) {
        checksNumber = atoi(argv[2]);
    }

    if (argc > 3) {
        popSize = atoi(argv[3]);
    }

    if (argc > 4) {
        genNumber = atoi(argv[4]);
    }

    if (argc > 5) {
        mutRate = atof(argv[5]);
    }

    if (argc > 6) {
        crossRate = atof(argv[6]);
    }

    // Create a field of checks.
    field = (cell_t *) malloc(fieldSize * fieldSize * sizeof(cell_t));
    checks = (PathGenome::_2DDot *) malloc(checksNumber * sizeof(PathGenome::_2DDot));

    for (unsigned int i = 0; i < fieldSize * fieldSize; i++) {
        field[i] = EMPTY;
    }

    srand(time(NULL));
    for (unsigned int i = 0; i < checksNumber; i++) {
        checks[i].x = (rand() % fieldSize);
        checks[i].y = (rand() % fieldSize);
        checks[i].id = i;
        field[IDX(checks[i].x, checks[i].y, fieldSize)] = true;
    }

    dump(field, NULL, fieldSize, checksNumber, "field.ppm");

    std::cout << "Field:\n";
    for (unsigned int i = 0; i < checksNumber; i++) {
        std::cout << "x:" << checks[i].x << "\ty:" << checks[i].y << "\n";
    }

    // Create a genome.
    PathGenome genome(checksNumber, checks);
    genome.evaluator(serialEvaluator);
    // PathGenome genome2(CHECKS_NUMBER, checks);

    // genome.initialize();
    // genome2.initialize();
    // std::cout << "parent1:\n" << genome << "\n";
    // std::cout << "parent2:\n" << genome2 << "\n";
    //
    // PathGenome *c = (PathGenome *) genome.clone(GAGenome::CONTENTS);
    // PathGenome::onePointCrossover(genome, genome2, c, 0);   // test single child crossover
    // std::cout << "child of crossover:\n" << *c << "\n";

    // for (int i = 0; i < 100; i++) {
    //     genome.initialize();
    //     char fileName[200];
    //     snprintf(fileName, 200, "path%d.ppm", i);
    //     dump(field, genome.getPath(), FIELD_SIZE, fileName);
    // }

    // Create a population.
    GAPopulation population(genome, popSize);
    population.selector(GARankSelector(GASelectionScheme::RAW));
    // population.evaluator(populationEvaluator);
    // population.evaluator(cudaEvaluator);
    // cudaEvaluator(population);

    // Create the genetic algorithm.
    GASimpleGA ga(population);
    // ga.populationSize(POP_SIZE);
    ga.nGenerations(genNumber);
    ga.pMutation(mutRate);
    ga.pCrossover(crossRate);
    // ga.set("el", gaFalse);
    // ga.selector(GARankSelector(GASelectionScheme::RAW));

    ga.initialize();
    // std::cout << "\nThe GA initialized the population, here it is:\n" << ga.population();

    ga.minimize();

    // for (int i = 0; i < ga.population().size(); i++) {
    //     printf("\nelement%d score:%f", i, ga.population().individual().);
    // }
    // printf("\ngen 1 best score:%f\n", ga.population().min());
    // printf("\ngen 0 best score:%f\tworst score:%f\n", ga.population().min(), ga.population().max());
    // for (int i = 0; i < ga.population().size(); i++) {
    //     printf("\ngen 1 individual %d score %f\n", i, ga.population().individual(i).score());
    //     // snprintf(fileName, 200, "element%d.ppm", i);
    //     // dump(field, ((PathGenome &) ga.population().individual(i)).getPath(), FIELD_SIZE, fileName);
    // }

    // for (int i = 0; i < ga.population().size(); i++) {
    //     char fileName[200];
    //     snprintf(fileName, 200, "element%d.ppm", i);
    //     dump(field, ((PathGenome &) ga.population().individual(i)).getPath(), FIELD_SIZE, fileName);
    // }
    PathGenome &bestOfAll = (PathGenome &) ga.population().best();
    // int bestGen = 0;
    snprintf(fileName, 200, "WorstOfGeneration%d.ppm", ga.generation());
    dump(field, ((PathGenome &) ga.population().worst()).getPath(), fieldSize, checksNumber, fileName);

    startTime = hpc_gettime();
    ga.evolve();
    endTime = hpc_gettime();
    printf("\ntime:%fs\n", endTime - startTime);
    // for (int i = 0; i < ga.nGenerations(); i++) {
    //     // getchar();
    //     // printf("\nGENERATION %d\n", ga.generation() + 1);
    //     // for (int j = 0; j < ga.population().size(); j++) {
    //     //     printf("\ngen %d individual %d score %f\n", i, j, ga.population().individual(i).score());
    //     //     // snprintf(fileName, 200, "element%d.ppm", i);
    //     //     // dump(field, ((PathGenome &) ga.population().individual(i)).getPath(), FIELD_SIZE, fileName);
    //     // }
    //     ga.step();
    //     // if (((PathGenome &) ga.population().best()).score() > bestOfAll.score()) {
    //     //     bestOfAll = ga.population().best();
    //     //     bestGen = ga.generation();
    //     // }
    //     printf("\ngeneration %d best score:%f\tworst score:%f\n", i, ga.population().min(), ga.population().max());
    //     // snprintf(fileName, 200, "WorstOfGeneration%d.ppm", i);
    //     // dump(field, ((PathGenome &) ga.population().worst()).getPath(), FIELD_SIZE, fileName);
    // }

    // snprintf(fileName, 200, "BestOfAll%d.ppm", bestGen);
    // dump(field, bestOfAll.getPath(), FIELD_SIZE, fileName);
    snprintf(fileName, 200, "BestOfGeneration%d.ppm", ga.generation());
    dump(field, ((PathGenome &) ga.population().best()).getPath(), fieldSize, checksNumber, fileName);

    std::cout << ga.statistics() << std::endl;

    return 0;
}
