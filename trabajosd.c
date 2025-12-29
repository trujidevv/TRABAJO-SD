#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <mpi.h>
#include <omp.h>

// Requirements: h=24. Evaluation on last 1000 lines.
#define HORIZON 24
#define TEST_SIZE 1000

typedef struct {
    int index;
    double dist;
} Neighbor;

// Compare function for qsort
int compareNeighbors(const void *a, const void *b) {
    Neighbor *n1 = (Neighbor *)a;
    Neighbor *n2 = (Neighbor *)b;
    if (n1->dist < n2->dist) return -1;
    if (n1->dist > n2->dist) return 1;
    return 0;
}

int main(int argc, char *argv[]) {
    int provided;
    // Initialization of MPI with thread support
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 3) {
        if (rank == 0) printf("Uso: %s <K> <archivo_datos>\n", argv[0]);
        MPI_Finalize();
        return 1;
    }

    int K = atoi(argv[1]);
    char *filename = argv[2];

    // --- LEER DATOS ---
    // According to instructions, files are on all machines, so standard fopen is acceptable for simplicity to load into memory
    FILE *fp = fopen(filename, "r");
    if (!fp) {
        if (rank == 0) perror("Error abriendo archivo");
        MPI_Finalize();
        return 1;
    }

    int rows, cols;
    // Format describes: First line with rows and cols
    if (fscanf(fp, "%d %d", &rows, &cols) != 2) {
        if (rank == 0) printf("Error leyendo cabecera\n");
        MPI_Finalize();
        return 1;
    }

    // Allocate full dataset
    float *data = (float *)malloc(rows * cols * sizeof(float));
    
    // Read CSV-like data
    for (int i = 0; i < rows * cols; i++) {
        fscanf(fp, "%f", &data[i]);
        fgetc(fp); // Consume delimiter (comma) or newline
    }
    fclose(fp);

    // Barrier to ensure all loaded before timing
    MPI_Barrier(MPI_COMM_WORLD);
    double start_time = MPI_Wtime();

    // --- DISTRIBUCIÓN ---
    // Evaluation only on the last 1000 rows
    int start_test_idx = rows - TEST_SIZE;
    if (start_test_idx < 0) start_test_idx = 0; // Fallback if file is small

    int total_tasks = rows - start_test_idx; // Should be 1000
    
    int tasks_per_proc = total_tasks / size;
    int remainder = total_tasks % size;

    int my_start_offset = rank * tasks_per_proc + (rank < remainder ? rank : remainder);
    int my_count = tasks_per_proc + (rank < remainder ? 1 : 0);

    // Buffers for local results
    float *my_predictions = (float *)malloc(my_count * cols * sizeof(float));
    double *my_mapes = (double *)malloc(my_count * sizeof(double));

    // --- PROCESAMIENTO ---
    for (int i = 0; i < my_count; i++) {
        int target_row = start_test_idx + my_start_offset + i;
        
        // Predict 'target_row' based on 'target_row - 1' (current day)
        // History: 0 ... target_row - 2
        int query_row = target_row - 1;
        int history_limit = query_row; // Rows 0 to history_limit-1 are candidates

        float *query_vec = &data[query_row * cols];
        Neighbor *candidates = (Neighbor *)malloc(history_limit * sizeof(Neighbor));

        // OpenMP Parallelization for Neighbor Search
        #pragma omp parallel for schedule(static)
        for (int h = 0; h < history_limit; h++) {
            float *hist_vec = &data[h * cols];
            double sum_sq = 0.0;
            for (int c = 0; c < cols; c++) {
                double diff = query_vec[c] - hist_vec[c];
                sum_sq += diff * diff;
            }
            candidates[h].index = h;
            candidates[h].dist = sqrt(sum_sq);
        }

        // Find K nearest neighbors
        // Using qsort (simple, O(N log N))
        qsort(candidates, history_limit, sizeof(Neighbor), compareNeighbors);

        // Calculate Average Prediction
        float prediction[HORIZON];
        for (int c = 0; c < cols; c++) prediction[c] = 0.0;

        for (int k = 0; k < K; k++) {
            int neighbor_idx = candidates[k].index;
            int next_day_idx = neighbor_idx + 1; // The day AFTER the neighbor
            
            float *next_vec = &data[next_day_idx * cols];
            for (int c = 0; c < cols; c++) {
                prediction[c] += next_vec[c];
            }
        }

        for (int c = 0; c < cols; c++) {
            prediction[c] /= K;
            my_predictions[i * cols + c] = prediction[c];
        }

        // Calculate MAPE for this row
        double row_mape_sum = 0.0;
        float *actual_vec = &data[target_row * cols];
        for (int c = 0; c < cols; c++) {
            if (fabs(actual_vec[c]) > 1e-9) {
                row_mape_sum += fabs((actual_vec[c] - prediction[c]) / actual_vec[c]);
            }
        }
        my_mapes[i] = (row_mape_sum * 100.0) / cols;

        free(candidates);
    }

    double end_time = MPI_Wtime();
    double my_time = end_time - start_time;
    double max_time;
    
    // Get max time among processes for the report
    MPI_Reduce(&my_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // Sum all MAPEs to calculate global average
    double my_mape_sum = 0.0;
    for(int i=0; i<my_count; i++) my_mape_sum += my_mapes[i];
    
    double global_mape_sum;
    MPI_Reduce(&my_mape_sum, &global_mape_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);


    // --- ESCRITURA DE SALIDA (USANDO MPI_File) ---
    MPI_Status status;

    // 1. Predicciones.txt
    // We remove the file first to ensure fresh creation
    if (rank == 0) remove("Predicciones.txt");
    MPI_Barrier(MPI_COMM_WORLD);

    MPI_File fh_pred;
    MPI_File_open(MPI_COMM_WORLD, "Predicciones.txt", MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fh_pred);
    
    // Build Buffer
    // Each float ~10 chars + separators. 24 floats -> ~300 chars.
    char *pred_buf = (char *)malloc(my_count * 500 * sizeof(char));
    int p_offset = 0;
    for (int i = 0; i < my_count; i++) {
        for (int c = 0; c < cols; c++) {
            // Using %.4f as simple precision format
            p_offset += sprintf(pred_buf + p_offset, "%.4f%c", my_predictions[i * cols + c], (c == cols - 1) ? '\n' : ' ');
        }
    }
    // Write ordered ensures the file is written in rank order (correct sequence of rows)
    MPI_File_write_ordered(fh_pred, pred_buf, p_offset, MPI_CHAR, &status);
    MPI_File_close(&fh_pred);
    free(pred_buf);

    // 2. MAPE.txt
    if (rank == 0) remove("MAPE.txt");
    MPI_Barrier(MPI_COMM_WORLD);

    MPI_File fh_mape;
    MPI_File_open(MPI_COMM_WORLD, "MAPE.txt", MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fh_mape);
    
    char *mape_buf = (char *)malloc(my_count * 50 * sizeof(char));
    int m_offset = 0;
    for (int i = 0; i < my_count; i++) {
        m_offset += sprintf(mape_buf + m_offset, "%.4f\n", my_mapes[i]);
    }
    MPI_File_write_ordered(fh_mape, mape_buf, m_offset, MPI_CHAR, &status);
    MPI_File_close(&fh_mape);
    free(mape_buf);

    // 3. Tiempo.txt (Written only by Rank 0)
    if (rank == 0) {
        FILE *ft = fopen("Tiempo.txt", "w");
        double avg_mape = global_mape_sum / total_tasks;
        fprintf(ft, "Tiempo ejecución: %.6f s\n", max_time);
        fprintf(ft, "Fichero: %s\n", filename);
        fprintf(ft, "MAPE Global: %.4f %%\n", avg_mape);
        fprintf(ft, "Procesos: %d\n", size);
        fprintf(ft, "Hilos: %d\n", omp_get_max_threads());
        fclose(ft);
    }

    free(my_predictions);
    free(my_mapes);
    free(data);

    MPI_Finalize();
    return 0;
}