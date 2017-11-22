#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include "../hpc.h"
#include "Population.h"
#include "PathGenome.h"

// Translates bidimensional indexes to a monodimensional one.
// |i| is the column index.
// |j| is the row index.
// |n| is the number of columns (length of the rows).
#define IDX(i, j, n) ((i) * (n) + (j))
#define BUFLEN 256

typedef unsigned char cell_t;

enum CellContent {
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

void dump(const cell_t *field, const PathGenome::_Point2D *path, unsigned int n, unsigned int checksNum, const char *filename) {
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


int main(int argc, char const *argv[]) {

    unsigned int fieldSize = 500;
    unsigned int checksNumber = 15;
    unsigned int popSize = 4096;
    unsigned int genNumber = 1000;
    float mutRate = 0.1;
    float crossRate = 1;

    // char fileName[200];
    double startTime = 0.0;
    double endTime = 0.0;
    char filename[BUFLEN];
    cell_t *field;
    PathGenome::_Point2D *checks;

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
    checks = (PathGenome::_Point2D *) malloc(checksNumber * sizeof(PathGenome::_Point2D));

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

    printf("Field:\n");
    for (unsigned int i = 0; i < checksNumber; i++) {
        printf("x:%d\ty:%d\n", checks[i].x, checks[i].y);
    }





    Population *population = new Population(popSize, genNumber, mutRate, new PathGenome(checks, checksNumber));
    population->initialize();
    population->sort();
    dump(field, ((PathGenome *) population->best())->path, fieldSize, checksNumber, "initial.ppm");

    printf("\nExecution:\n");
    startTime = hpc_gettime();
    for (unsigned int i = 0; i < genNumber; i++) {
        // printf("Generation %u\n", i);
        population->step();
    }
    endTime = hpc_gettime();
    printf("Execution time (s):%f\n\n", endTime - startTime);

    // char *string = (char *) malloc(checksNumber * POINT_SIZE);
    // population->outputBest(string);
    // outputBest<<<1, d_checksNum>>>(d_pop, d_string);
    //
    // for (unsigned int i = 0; i < checksNumber; i++) {
    //     memcpy(&(checks[i].x), &(string[i * POINT_SIZE]), COORD_SIZE);
    //     memcpy(&(checks[i].y), &(string[i * POINT_SIZE + COORD_SIZE]), COORD_SIZE);
    //     printf("x:%d\ty:%d\n", checks[i].x, checks[i].y);
    // }

    dump(field, ((PathGenome *) population->best())->path, fieldSize, checksNumber, "final.ppm");


    free(field);

    return 0;
}
