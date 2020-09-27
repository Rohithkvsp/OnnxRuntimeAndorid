//
//  onnxruntime_inference.hpp
//  onnxRuntimeInference
//
//  Created by rohith kvsp on 9/13/20.
//  Copyright © 2020 rohith kvsp. All rights reserved.
//

#ifndef onnxruntime_inference_hpp
#define onnxruntime_inference_hpp

#include <stdio.h>
#include <chrono>
#include <cinttypes>
#include "onnxruntime/core/session/onnxruntime_cxx_api.h"
#include "onnxruntime_c_api.h"
#include "onnxruntime/core/providers/nnapi/nnapi_provider_factory.h"



class Inference {
    
private:
//    Ort::Env& env_;
//    Ort::Session session{nullptr};

    std::unique_ptr<Ort::Env> env_;
    std::unique_ptr<Ort::Session> session_;

    const char* modelpath_;
    const char* labelfilepath_;
    int img_height_;
    int img_width_;

    
    unsigned long input_tensor_size;
    unsigned long ouput_tensor_size;
    
    std::vector<int64_t> input_node_dims;
    std::vector<int64_t> output_node_dims;
    
    Ort::Value input_tensor{nullptr};
    std::vector<const char*> input_node_names;
    std::vector<const char*> output_node_names;
    
    std::unique_ptr<float[]> input_data_chw;
    std::unique_ptr<float[]> normalized;
    std::unique_ptr<float[]> softmax_output;
    
    std::vector<std::string> labels;
    std::string predicted_labels;


    
    void createInputBuffer();
    void printNodes();



public:

    Inference(Ort::Env* env,  const char*  modelpath, const char*  labelfilepath,  int img_height, int img_width);
    Inference(const Inference& ) = delete; //no copy
    Inference& operator = (const Inference &) = delete;//no copy
    void run(uint8_t* pixels);
    std::string getPredictedlabels();
  
    ~Inference();
    
};

#endif /* onnxruntime_inference_hpp */
