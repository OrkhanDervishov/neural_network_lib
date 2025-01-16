#define MATRIX_IMPLEMENTATION
#define NN_IMPLEMENTATION
#define AUTO_NN_FUNCS_IMPLEMENTATIONS
#include "nn.h"

#include <time.h>

// float* and_data[] = 
// {
//     {1, 1},
//     {1, 0},
//     {0, 1},
//     {0, 0}
// };

float and_labels[] = {1, 0, 0, 0};


int main()
{

    float** and_data1 = (float**)malloc(sizeof(float*) * 4);
    for(size_t i = 0; i < 4; i++)
    {
        and_data1[i] = (float*)malloc(sizeof(float) * 2);
    }

    and_data1[0][0] = 1;
    and_data1[0][1] = 1;
    
    and_data1[1][0] = 1;
    and_data1[1][1] = 0;
    
    and_data1[2][0] = 0;
    and_data1[2][1] = 1;
    
    and_data1[3][0] = 0;
    and_data1[3][1] = 0;


    float** and_labels1 = (float**)malloc(sizeof(float*) * 4);
    for(size_t i = 0; i < 4; i++)
    {
        and_labels1[i] = (float*)malloc(sizeof(float) * 1);
    }

    and_labels1[0][0] = 1;
    
    and_labels1[1][0] = 0;
    
    and_labels1[2][0] = 0;
    
    and_labels1[3][0] = 0;




    //srand(time(0));
    size_t layers = 4;
    size_t neuron_layers[] = {20, 100, 40 ,10};

    Model m = auto_create_model(layers, neuron_layers, 0);

    return 0;
    Dataset ds = auto_create_dataset(4, 2, 1, and_data1, and_labels1);

    //mat_print(ds.labels[0]);

    printf("cost = %f\n", cost(m, ds));

    train(m, ds, 100000);

    printf("cost = %f\n", cost(m, ds));





    // INPUT_LAYER(m) = input;

    // forward(m);

    // mat_print(OUTPUT_LAYER(m));

    auto_destroy_dataset(ds);
    auto_destroy_model(m);

    return 0;
}