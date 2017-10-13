#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <ga/ga.h>
#include "PathGenome.h"

// Translates bidimensional indexes to a monodimensional one.
// |i| is the column index.
// |j| is the row index.
// |n| is the number of columns (length of the rows).
#define IDX(i, j, n) ((i) * (n) + (j))

#define POP_SIZE 10
#define GEN_NUMBER 5
#define MUT_RATE 0.5
#define CROSS_RATE 1

#define FIELD_SIZE 300
#define CHECKS_NUMBER 10

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

void dump(const cell_t *field, const PathGenome::_2DDot *path, int n, const char *filename) {

    cell_t *fieldCopy = (cell_t *) malloc(n * n * sizeof(cell_t));
    for (int x = 0; x < n; x++) {
        for (int y = 0; y < n; y++) {
            fieldCopy[IDX(x, y, n)] = field[IDX(x, y, n)];
        }
    }

    if (path != NULL) {
        for (int i = 0; i < CHECKS_NUMBER; i++) {
            drawLine(fieldCopy, n, path[i].x, path[i].y, path[(i + 1) % CHECKS_NUMBER].x, path[(i + 1) % CHECKS_NUMBER].y);
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
    for (int x = 0; x < n; x++) {
        for (int y = 0; y < n; y++) {
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




float fitness(GAGenome &g) {
    GA1DBinaryStringGenome &genome = (GA1DBinaryStringGenome &)g;

    float score=0.0;
    for (int i = 0; i < genome.length(); i++) {
        // The more 1s are contained in the string, the higher is the fitness.
        // The score is incremented by the value of the current element of the string (0 or 1).
        score += genome.gene(i);
    }
    return score;
}




void cudaEvaluator(GAPopulation &p) {
    dim3 blockSize(CHECKS_NUMBER);
    for (int i = 0; i < p.size(); i++) {
        GAGenome *individual = &(p.individual(i));
        // Allocate memory for the genome object on the device.
        GAGenome *d_individual;
        cudaMalloc(&d_individual, sizeof(GAGenome));
        // Copy the genome object to the device.
        cudaMemcpy(d_individual, &individual, sizeof(GAGenome), cudaMemcpyHostToDevice);

        // Allocate memory for the genome object's pointers on the device.
        // PathGenome::_2DDot *d_checks;
        PathGenome *ind = (PathGenome *) individual;
        PathGenome::_2DDot *d_path;
        float *d_distances;
        // cudaMalloc(&d_checks, sizeof(PathGenome::_2DDot));
        cudaMalloc(&d_path, sizeof(PathGenome::_2DDot));
        cudaMalloc(&d_distances, sizeof(float));
        // Copy the genome object' pointers on the device.
        cudaMemcpy(d_path, ind->getPath(), sizeof(PathGenome::_2DDot), cudaMemcpyHostToDevice);
        cudaMemcpy(d_distances, ind->getDistances(), sizeof(float), cudaMemcpyHostToDevice);

        // Set the score.
    }
}




float serialEvaluator(GAGenome &g) {
    PathGenome &genome = (PathGenome &) g;
    float distance = 0.0;
    float score = 0.0;
    for (unsigned int i = 0; i < genome.getChecksNum(); i++) {
        float dx = (float) genome.gene((i + 1) % genome.getChecksNum()).x - (float) genome.gene(i).x;
        float dy = (float) genome.gene((i + 1) % genome.getChecksNum()).y - (float) genome.gene(i).y;
        distance = sqrtf(powf(dx, 2) + powf(dy, 2));
        genome.setDistance(i, distance);
    }
    for (unsigned int i = 0; i < genome.getChecksNum(); i++) {
        // printf("distance %d on host:%f\n\n", i, genome->getDistances()[i]);
        score += genome.getDistances()[i];
    }
    return score;
}




int main(int argc, char const *argv[]) {
    // Create a field of checks.
    cell_t *field;
    PathGenome::_2DDot *checks;

    field = (cell_t *) malloc(FIELD_SIZE * FIELD_SIZE * sizeof(cell_t));
    checks = (PathGenome::_2DDot *) malloc(CHECKS_NUMBER * sizeof(PathGenome::_2DDot));

    for (int i = 0; i < FIELD_SIZE * FIELD_SIZE; i++) {
        field[i] = EMPTY;
    }

    srand(time(NULL));
    for (int i = 0; i < CHECKS_NUMBER; i++) {
        checks[i].x = (rand() % FIELD_SIZE);
        checks[i].y = (rand() % FIELD_SIZE);
        checks[i].id = i;
        field[IDX(checks[i].x, checks[i].y, FIELD_SIZE)] = true;
    }

    dump(field, NULL, FIELD_SIZE, "field.ppm");

    std::cout << "Field:\n";
    for (int i = 0; i < CHECKS_NUMBER; i++) {
        std::cout << "x:" << checks[i].x << "\ty:" << checks[i].y << "\n";
    }

    // Create a genome.
    PathGenome genome(CHECKS_NUMBER, checks);
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
    // GAPopulation population(genome, POP_SIZE);
    // population.evaluator(cudaEvaluator);
    // cudaEvaluator(population);

    // Create the genetic algorithm.
    GASimpleGA ga(genome);
    ga.populationSize(POP_SIZE);
    ga.nGenerations(GEN_NUMBER);
    ga.pMutation(MUT_RATE);
    ga.pCrossover(CROSS_RATE);
    ga.selector(GARankSelector(GASelectionScheme::RAW));

    ga.initialize();
    // std::cout << "\nThe GA initialized the population, here it is:\n" << ga.population();

    ga.minimize();

    char fileName[200];
    // for (int i = 0; i < ga.population().size(); i++) {
    //     printf("\nelement%d score:%f", i, ga.population().individual().);
    // }
    // printf("\ngen 1 best score:%f\n", ga.population().min());
    printf("\ngen 0 best score:%f\tworst score:%f\n", ga.population().min(), ga.population().max());
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
    snprintf(fileName, 200, "BestOfGeneration%d.ppm", ga.generation());
    dump(field, ((PathGenome &) ga.population().best()).getPath(), FIELD_SIZE, fileName);

    for (int i = 0; i < ga.nGenerations(); i++) {
        getchar();
        // printf("\nGENERATION %d\n", ga.generation() + 1);
        // for (int j = 0; j < ga.population().size(); j++) {
        //     printf("\ngen %d individual %d score %f\n", i, j, ga.population().individual(i).score());
        //     // snprintf(fileName, 200, "element%d.ppm", i);
        //     // dump(field, ((PathGenome &) ga.population().individual(i)).getPath(), FIELD_SIZE, fileName);
        // }
        ga.step();
        printf("\ngeneration %d best score:%f\tworst score:%f\n", i, ga.population().min(), ga.population().max());
        // snprintf(fileName, 200, "WorstOfGeneration%d.ppm", i);
        // dump(field, ((PathGenome &) ga.population().worst()).getPath(), FIELD_SIZE, fileName);
    }

    // snprintf(fileName, 200, "BestOfGeneration%d.ppm", ga.generation());
    // dump(field, ((PathGenome &) ga.population().best()).getPath(), FIELD_SIZE, fileName);

    std::cout << ga.statistics() << std::endl;

    return 0;
}
