//
//  postprocess.h
//  onnxRuntimeInference
//
//  Created by rohith kvsp on 9/13/20.
//  Copyright © 2020 rohith kvsp. All rights reserved.
//

#ifndef postprocess_h
#define postprocess_h

#include <vector>
#include <queue>
#include <limits>
#include <cmath>

/**
 * Method sorts probabilty values and returns k top scores with index as vector
 * @param values : sofmax probabilty value pointer;
 * @param len : length of the pointer
 * @param k : k values
 * @param results : vector of probabilty value and index
 */

void KTopResults(float *values, int len, int k, std::vector<std::pair<float, int>>& results)
{
    std::priority_queue<std::pair<float,int>, std::vector<std::pair<float, int>>,
    std::greater<std::pair<float, int>>> min_heap;
    
    for(int i=0; i<len; i++)
    {
        min_heap.push({values[i],i});
        
        if(min_heap.size()>k)
            min_heap.pop();
    }
    
    while(!min_heap.empty())
    {
        results.push_back(min_heap.top());
        min_heap.pop();
    }
    std::reverse(results.begin(), results.end());
    
    return;

}
/**
 * Softmax layer to produce probabilty values
 * @param input_value : input floating array pointer
 * @param output_value : output floating array pointer
 * @param in_out_size : size of input, output pointer
 */

void softmax(float *input_value, float* output_value, int in_out_size )
{
    
    
    float max_val = std::numeric_limits<float>::lowest();
    for(int i = 0; i < in_out_size; i++) {
        max_val = std::max(input_value[i], max_val);
    }
    
    float sum  = 0.0f;
    for(int i = 0; i < in_out_size; i++){
        float exp_val = std::exp(input_value[i] - max_val);
        output_value[i] = exp_val;
        sum += exp_val;
    }
    
    for(int i = 0; i < in_out_size; i++) {
        output_value[i] = output_value[i]/sum;
    }
    
    return;
    
}

#endif /* postprocess_h */
