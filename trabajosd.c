#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <mpi.h>
#include <omp.h>

#define HORIZONTE 24 // h=24 según enunciado [cite: 21]
#define TEST_SIZE 1000 // Evaluar sobre las últimas 1000 filas [cite: 68]

// Estructura para almacenar vecinos y sus distancias
typedef struct {
    int index;
    double distance;
} Neighbor;

// Función para calcular distancia Euclidea (OpenMP se usará en el bucle que llama a esto)
double calcular_distancia(float *a, float *b, int h) {
    double sum = 0.0;
    for (int i = 0; i < h; i++) {
        double diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sqrt(sum);
}

// Función para comparar vecinos (para ordenar qsort)
int compare_neighbors(const void *a, const void *b) {
    Neighbor *n1 = (Neighbor *)a;
    Neighbor *n2 = (Neighbor *)b;
    if (n1->distance < n2->distance) return -1;
    if (n1->distance > n2->distance) return 1;
    return 0;
}

int main(int argc, char *argv[]) {
    int rank, size;
    int provided;

    // Inicialización MPI con soporte para hilos
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 3) {
        if (rank == 0) printf("Uso: %s <K> <archivo_datos>\n", argv[0]);
        MPI_Finalize();
        return 1;
    }

    int K = atoi(argv[1]); [cite: 51]
    char *filename = argv[2]; [cite: 52]

    int rows, cols;
    float *dataset = NULL;

    // --- LECTURA DE DATOS ---
    // Por simplicidad y dado que los archivos están en todas las máquinas[cite: 76],
    // cada proceso lee el archivo. Para archivos gigantes, se usaría MPI_File_read.
    FILE *fp = fopen(filename, "r");
    if (!fp) {
        perror("Error abriendo archivo");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    fscanf(fp, "%d %d", &rows, &cols); // Leer cabecera

    // Reservar memoria para TODO el dataset (necesario para buscar vecinos)
    // Se almacena como un array 1D contiguo
    dataset = (float *)malloc(rows * cols * sizeof(float));

    // Leer datos (separados por comas)
    for (int i = 0; i < rows * cols; i++) {
        float val;
        // Truco para saltar comas: leer float y char separador
        if (i < rows * cols - 1) fscanf(fp, "%f,", &dataset[i]);
        else fscanf(fp, "%f", &dataset[i]);
    }
    fclose(fp);

    // Validar que tenemos suficientes datos
    if (rows <= TEST_SIZE) {
        if (rank == 0) printf("Error: El dataset debe tener más de %d filas.\n", TEST_SIZE);
        MPI_Finalize();
        return 1;
    }

    double start_time = MPI_Wtime();

    // --- DISTRIBUCIÓN DE TRABAJO ---
    // Solo predecimos las últimas 1000 filas [cite: 68]
    int start_test_index = rows - TEST_SIZE; 
    
    // Repartir las 1000 tareas entre los procesos MPI
    int local_n = TEST_SIZE / size;
    int remainder = TEST_SIZE % size;
    
    int my_start = start_test_index + rank * local_n + (rank < remainder ? rank : remainder);
    int my_count = local_n + (rank < remainder ? 1 : 0);

    // Arrays para guardar resultados locales
    float *local_predictions = (float *)malloc(my_count * cols * sizeof(float));
    double *local_mapes = (double *)malloc(my_count * sizeof(double));
    double local_mape_sum = 0.0;

    // --- BUCLE PRINCIPAL DE PREDICCIÓN ---
    for (int i = 0; i < my_count; i++) {
        int target_row_idx = my_start + i; // Índice de la fila que queremos predecir (R_n)
        int query_row_idx = target_row_idx - 1; // Usamos el día anterior para buscar parecidos

        // Puntero al vector del "día actual" (query)
        float *query_vec = &dataset[query_row_idx * cols];

        // Buscar en el histórico (desde fila 0 hasta query_row_idx - 1)
        // Necesitamos encontrar K vecinos.
        // Como el histórico es grande, el array de vecinos candidatos es grande.
        int history_size = query_row_idx - 1; // No podemos mirar al futuro ni al propio día
        
        Neighbor *all_neighbors = (Neighbor *)malloc(history_size * sizeof(Neighbor));

        // PARALELIZACIÓN OPENMP: Búsqueda de vecinos [cite: 16, 62]
        #pragma omp parallel for schedule(static)
        for (int h = 0; h < history_size; h++) {
            float *hist_vec = &dataset[h * cols];
            double dist = calcular_distancia(query_vec, hist_vec, cols);
            all_neighbors[h].index = h;
            all_neighbors[h].distance = dist;
        }

        // Ordenar para encontrar los K más cercanos (se puede optimizar con heap, pero qsort es simple)
        qsort(all_neighbors, history_size, sizeof(Neighbor), compare_neighbors);

        // Calcular predicción (promedio de los días siguientes a los vecinos) [cite: 24]
        float prediction[HORIZONTE];
        for (int d = 0; d < cols; d++) prediction[d] = 0.0;

        for (int k = 0; k < K; k++) {
            int neighbor_next_day_idx = all_neighbors[k].index + 1;
            float *next_val_vec = &dataset[neighbor_next_day_idx * cols];
            for (int d = 0; d < cols; d++) {
                prediction[d] += next_val_vec[d];
            }
        }

        // Dividir por K para obtener la media
        for (int d = 0; d < cols; d++) {
            prediction[d] /= K;
            // Guardar en array local
            local_predictions[i * cols + d] = prediction[d];
        }

        // Calcular MAPE de esta predicción [cite: 45]
        // Comparamos prediction[] contra el valor real dataset[target_row_idx]
        double row_mape = 0.0;
        float *real_val = &dataset[target_row_idx * cols];
        
        for (int d = 0; d < cols; d++) {
            if (fabs(real_val[d]) > 1e-9) { // Evitar división por cero
                row_mape += fabs((real_val[d] - prediction[d]) / real_val[d]);
            }
        }
        // Fórmula del enunciado: (100/h) * sum(...)
        local_mapes[i] = (100.0 / cols) * row_mape;
        local_mape_sum += local_mapes[i];

        free(all_neighbors);
    }

    double end_time = MPI_Wtime();
    double total_time = end_time - start_time;
    double global_mape_sum = 0.0;

    // Reducir la suma de MAPE para obtener el promedio global
    MPI_Reduce(&local_mape_sum, &global_mape_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    // --- ESCRITURA DE RESULTADOS ---
    // Se pide generar Predicciones.txt, MAPE.txt y Tiempo.txt [cite: 57]
    // Usaremos MPI I/O o recolección en Rank 0. Por simplicidad, Rank 0 recoge y escribe.
    
    // Recolectar predicciones y MAPEs en Rank 0 es costoso en memoria si es gigante,
    // pero para 1000 filas x 24 cols es manejable (aprox 96KB).

    // Nota: gather con tamaños variables (gatherv) es necesario si la división no es exacta.
    // Para simplificar el ejemplo, asumiremos que Rank 0 escribe sus cosas y usa MPI_File o bucles de envio/recepcion
    // Aquí implemento una escritura secuencial ordenada usando tokens para evitar corrupción de archivos.
    
    // 1. Predicciones.txt
    // Usar MPI_File_open / write_ordered sería lo ideal según[cite: 77].
    MPI_File fh_pred, fh_mape;
    MPI_Status status;
    
    MPI_File_open(MPI_COMM_WORLD, "Predicciones.txt", MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fh_pred);
    
    // Convertir floats a texto es complicado con MPI_File directo binario. 
    // Escribiremos bloque a bloque convirtiendo a string.
    char *chunk_pred_str = (char *)malloc(my_count * cols * 20 * sizeof(char)); // Buffer aprox
    chunk_pred_str[0] = '\0';
    for(int i=0; i<my_count; i++) {
        for(int j=0; j<cols; j++) {
            char buffer[32];
            sprintf(buffer, "%.4f%s", local_predictions[i*cols + j], (j==cols-1) ? "\n" : ", ");
            strcat(chunk_pred_str, buffer);
        }
    }
    MPI_File_write_ordered(fh_pred, chunk_pred_str, strlen(chunk_pred_str), MPI_CHAR, &status);
    MPI_File_close(&fh_pred);
    free(chunk_pred_str);

    // 2. MAPE.txt
    MPI_File_open(MPI_COMM_WORLD, "MAPE.txt", MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fh_mape);
    char *chunk_mape_str = (char *)malloc(my_count * 20 * sizeof(char));
    chunk_mape_str[0] = '\0';
    for(int i=0; i<my_count; i++) {
        char buffer[32];
        sprintf(buffer, "%.4f\n", local_mapes[i]);
        strcat(chunk_mape_str, buffer);
    }
    MPI_File_write_ordered(fh_mape, chunk_mape_str, strlen(chunk_mape_str), MPI_CHAR, &status);
    MPI_File_close(&fh_mape);
    free(chunk_mape_str);

    // 3. Tiempo.txt (Solo rank 0) [cite: 60]
    if (rank == 0) {
        FILE *f_time = fopen("Tiempo.txt", "w");
        double avg_mape = global_mape_sum / TEST_SIZE;
        // Obtener número de hilos máximo
        int max_threads = omp_get_max_threads();
        fprintf(f_time, "Tiempo: %f s\nArchivo: %s\nMAPE Global: %f\nProcesos: %d\nHilos: %d\n", 
                total_time, filename, avg_mape, size, max_threads);
        fclose(f_time);
        printf("Procesamiento completado en %f s. MAPE global: %f\n", total_time, avg_mape);
    }

    free(dataset);
    free(local_predictions);
    free(local_mapes);

    MPI_Finalize();
    return 0;
}