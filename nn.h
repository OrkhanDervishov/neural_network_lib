
/*NN matrix functions*/

#ifndef MATRIX_H_
#define MATRIX_H_

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <malloc.h>
#include <math.h>

#define PARAM_TYPE float

// Matrix Declarations
/*************************************************************** */

typedef struct{
    size_t w;
    size_t h;
    PARAM_TYPE *elems;
} Mat;

typedef struct{
    size_t l;
    PARAM_TYPE *elems;
} Vec;

#define MAT_LEN(m) sizeof((m).elems) / sizeof((m).elems[0])

#define MAT_INDEX(m, i, j) (m).elems[(i)*(m).w + (j)]


float rand_float();

Mat gen_mat(size_t h, size_t w);

void free_mat(Mat m);

void mat_dot(Mat dest, Mat a, Mat b);

void mat_sum(Mat dest, Mat m);

void mat_cpy(Mat dest, Mat src);

void transpose(Mat dest, Mat m);

void mat_fill(Mat m, PARAM_TYPE value);

void fill_rand(Mat m, float high, float low);

void mat_print(Mat m);

#endif //MATRIX_H_


//************************************************** */
// IMPLEMENTATIONS

#ifdef MATRIX_IMPLEMENTATION

float rand_float()
{
    return rand() / 30000.0f;
}

Mat gen_mat(size_t h, size_t w)
{
    Mat m = {m.w = w,   m.h = h,    m.elems = (PARAM_TYPE*)malloc(sizeof(PARAM_TYPE) * w *h)};
    return m;
}

void free_mat(Mat m)
{
    free(m.elems);
}

float sigmoid(float y)
{
    return 1.0f / (1.0f + expf(y));
}

void mat_sig(Mat m)
{

    for(size_t i = 0; i < m.h; i++)
    {
        for(size_t j = 0; j < m.w; j++)
        {
            MAT_INDEX(m, i, j) = sigmoid(MAT_INDEX(m, i, j));
        }
    }
}

void mat_dot(Mat dest, Mat a, Mat b)
{
    // 1x2 2x3

    assert(a.w == b.h);
    assert(dest.h == a.h);
    assert(dest.w == b.w);

    size_t n = a.w;
    float sum = 0.0f;

    for(size_t i = 0; i < dest.h; i++)
    {
        for(size_t j = 0; j < dest.w; j++)
        {
            for(size_t k = 0; k < n; k++)
            {
                sum += MAT_INDEX(a, i, k) * MAT_INDEX(b, k, j);
            }
            MAT_INDEX(dest, i, j) = sum; 
        }
    }    
}

void mat_sum(Mat dest, Mat m)
{
    assert(dest.w == m.w);
    assert(dest.h == m.h);

    for(size_t i = 0; i < m.h; i++)
    {
        for(size_t j = 0; j < m.w; j++)
        {
            MAT_INDEX(dest, i, j) += MAT_INDEX(m, i, j); 
        }
    }
}

void mat_sub(Mat dest, Mat m)
{
    assert(dest.w == m.w);
    assert(dest.h == m.h);

    for(size_t i = 0; i < m.h; i++)
    {
        for(size_t j = 0; j < m.w; j++)
        {
            MAT_INDEX(dest, i, j) -= MAT_INDEX(m, i, j); 
        }
    }
}

void mat_cpy(Mat dest, Mat src)
{
    assert(dest.h == src.h);
    assert(dest.w == src.w);

    for(size_t i = 0; i < dest.h; i++)
    {
        for(size_t j = 0; j < dest.w; j++)
        {
            MAT_INDEX(dest, i, j) = MAT_INDEX(src, i, j);
        }
    }
}

void transpose(Mat dest, Mat m)
{
    assert(dest.h == m.w);
    assert(dest.w == m.h);

    for(size_t i = 0; i < m.h; i++)
    {
        for(size_t j = 0; j < m.w; j++)
        {
            MAT_INDEX(dest, j, i) = MAT_INDEX(m, i, j);
        }
    }
}


void fill(Mat m, PARAM_TYPE value)
{
    for(size_t i = 0; i < m.h; i++)
    {
        for(size_t j = 0; j < m.w; j++)
        {
            MAT_INDEX(m, i, j) = value; 
        }
    }
}

void fill_rand(Mat m, float high, float low)
{
    for(size_t i = 0; i < m.h; i++)
    {
        for(size_t j = 0; j < m.w; j++)
        {
            MAT_INDEX(m, i, j) = rand_float()*(high - low) + low; 
        }
    }
}

void mat_print(Mat m)
{
    for(size_t i = 0; i < m.h; i++)
    {
        for(size_t j = 0; j < m.w; j++)
        {
            printf("%f ", MAT_INDEX(m, i, j)); 
        }
        printf("\n");
    }
}

#endif //MATRIX_IMPLEMENTATION






/******************************************************************************** */

#ifndef NN_H_
#define NN_H_

#include "../math/matrix.h"


// NN Declarations
/*************************************************************** */

typedef struct{
    size_t layer_count;
    Mat *layers;
    Mat *weights;
    Mat *biases;
} Model;

#define INPUT_LAYER(m) ((m).layers[0])
#define OUTPUT_LAYER(m) ((m).layers[(m).layer_count - 1])



typedef struct{
    size_t data_count;
    Mat *data;
    Mat *labels;
} Dataset;


Model gen_model(size_t layer_count, Mat *layers, Mat *weights, Mat *biases);

Dataset gen_dataset(size_t data_count, Mat* data, Mat* labels);

float cost(Model m, Dataset ds);

void train_weights(Model m, Dataset ds, float eps, float rate);

void train_biases(Model m, Dataset ds, float eps, float rate);

void train(Model m, Dataset ds, size_t iter);

void forward(Model model);

#endif //NN_H_


/***************************************************/

#ifdef NN_IMPLEMENTATION

Model gen_model(size_t layer_count, Mat *layers, Mat *weights, Mat *biases)
{
    Model model = {layer_count,     layers,     weights,    biases};
    return model;
}


Dataset gen_dataset(size_t data_count, Mat* data, Mat* labels)
{
    Dataset dataset = {data_count,      data,       labels};
    return dataset;
}


float cost(Model m, Dataset ds)
{
    float cost = 0;
    Mat diff = gen_mat(ds.labels[0].h, ds.labels[0].w);

    for(size_t i = 0; i < ds.data_count; i++)
    {
        mat_cpy(m.layers[0], ds.data[i]);
        forward(m);
        mat_cpy(diff, m.layers[m.layer_count - 1]);
        mat_sub(diff, ds.labels[i]);

        for(size_t j = 0; j < diff.w; j++)
        {
            cost += MAT_INDEX(diff, 0, j) * MAT_INDEX(diff, 0, j);
        }
        cost = cost / diff.w;
    }

    free_mat(diff);
    return cost;
}


void train_weights(Model m, Dataset ds, float eps, float rate)
{
    float param = 0;

    for(size_t k = 0; k < m.layer_count - 1; k++)
    {
        for(size_t i = 0; i < m.weights[k].h; i++)
        {   
            for(size_t j = 0; j < m.weights[k].w; j++)
            {
                param = MAT_INDEX(m.weights[k], i, j);
                MAT_INDEX(m.weights[k], i, j) += eps;
                float ch = cost(m, ds);

                MAT_INDEX(m.weights[k], i, j) = param;
                float c = cost(m, ds);

                float grad = (ch - c) / eps;
                MAT_INDEX(m.weights[k], i, j) -= rate * grad;
            }
        }
    }
}


void train_biases(Model m, Dataset ds, float eps, float rate)
{
    float param;

    for(size_t k = 0; k < m.layer_count - 1; k++)
    {
        for(size_t j = 0; j < m.biases[k].w; j++)
        {
            param = MAT_INDEX(m.biases[k], 0, j);
            MAT_INDEX(m.biases[k], 0, j) += eps;
            float ch = cost(m, ds);

            MAT_INDEX(m.biases[k], 0, j) = param;
            float c = cost(m, ds);

            float grad = (ch - c) / eps;
            MAT_INDEX(m.biases[k], 0, j) -= rate * grad;
        }
    }
}

void train(Model m, Dataset ds, size_t iter)
{

    float eps = 1e-1;
    float rate = 1e-1;

    // train weight
    for(size_t i = 0; i < iter; i++)
    {
        train_biases(m, ds, eps, rate);
        train_weights(m, ds, eps, rate);
    }
}


void forward(Model m)
{

    for(size_t i = 1; i < m.layer_count; i++)
    {
        mat_dot(m.layers[i], m.layers[i - 1], m.weights[i - 1]);
        mat_sum(m.layers[i], m.biases[i - 1]);
        mat_sig(m.layers[i]);
    }
}

#endif // NN_IMPLEMENTATION




/**************************************************************************** */

#ifndef AUTO_NN_FUNCS_H_
#define AUTO_NN_FUNCS_H_


Model auto_create_model(size_t layer_count, size_t* layer_neurons, int rand);

int auto_destroy_model(Model m);

Dataset auto_create_dataset(size_t data_count, size_t input_layer_count, size_t output_layer_count, float** data_arr, float** labels_arr);

int auto_destroy_dataset(Dataset ds);


#ifdef AUTO_NN_FUNCS_IMPLEMENTATIONS


Model auto_create_model(size_t layer_count, size_t* layer_neurons, int rand)
{
    Model m;
    Mat* layers = (Mat*)malloc(sizeof(Mat) * layer_count);
    Mat* weights = (Mat*)malloc(sizeof(Mat) * (layer_count - 1));
    Mat* biases = (Mat*)malloc(sizeof(Mat) * (layer_count - 1));

    // Create layers
    for(size_t i = 0; i < layer_count; i++)
    {
        layers[i] = gen_mat(1, layer_neurons[i]);
    }

    //printf("\n\n\n");


    // Create weights
    for(size_t i = 0; i < layer_count - 1; i++)
    {
        weights[i] = gen_mat(layer_neurons[i], layer_neurons[i + 1]);

        if(rand){
            fill_rand(weights[i], 10.0f, 0.0f);

            printf("----------------\n");
            mat_print(weights[i]);
            continue;
        }
        fill(weights[i], 1);
    }

    //printf("\n\n\n");

    // Create biases
    for(size_t i = 0; i < layer_count - 1; i++)
    {
        biases[i] = gen_mat(1, layer_neurons[i + 1]);

        if(rand){
            fill_rand(biases[i], 10.0f, 0.0f);

            printf("----------------\n");
            mat_print(biases[i]);
            continue;
        }
        fill(biases[i], 1);
    }

    //printf("\n\n\n");

    m = gen_model(layer_count, layers, weights, biases);
    return m;
}

int auto_destroy_model(Model m)
{

    // Destroy layers
    for(size_t i = 0; i < m.layer_count; i++)
    {
        free_mat(m.layers[i]);
    }

    // Destroy weights
    for(size_t i = 0; i < m.layer_count - 1; i++)
    {
        free_mat(m.weights[i]);
    }

    // Destroy biases
    for(size_t i = 0; i < m.layer_count - 1; i++)
    {
        free_mat(m.biases[i]);
    }

    return 0;
}



Dataset auto_create_dataset(size_t data_count, size_t input_layer_count, size_t output_layer_count, float** data_arr, float** labels_arr)
{
    Mat* data = (Mat*)malloc(sizeof(Mat) * data_count);
    Mat* labels = (Mat*)malloc(sizeof(Mat) * data_count);

    for(size_t i = 0; i < data_count; i++)
    {
        data[i] = gen_mat(1, input_layer_count);

        for(size_t j = 0; j < input_layer_count; j++)
        {
            MAT_INDEX(data[i], 0, j) = data_arr[i][j];
        }
    }


    for(size_t i = 0; i < data_count; i++)
    {
        labels[i] = gen_mat(1, output_layer_count);

        for(size_t j = 0; j < output_layer_count; j++)
        {
            MAT_INDEX(labels[i], 0, j) = labels_arr[i][j];
        }
    }

    Dataset ds = gen_dataset(data_count, data, labels);  

    return ds;
}


int auto_destroy_dataset(Dataset ds)
{

    for(size_t i = 0; i < ds.data_count; i++)
    {
        free_mat(ds.data[i]);
    }
    free(ds.data);
    

    for(size_t i = 0; i < ds.data_count; i++)
    {
        free_mat(ds.labels[i]);
    }
    free(ds.labels);

    return 1;
}


#endif // NN_FUNCS_IMPLEMENTATIONS
#endif // GENERAL_NN_FUNCS_H_