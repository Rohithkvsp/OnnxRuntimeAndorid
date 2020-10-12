//
//  utils.h
//  onnxRuntimeInference
//
//  Created by rohith kvsp on 9/13/20.
//  Copyright © 2020 rohith kvsp. All rights reserved.
//

#ifndef utils_h
#define utils_h
#include <fstream>

/**
 * read label file and return labels as vector of string
 * @param file_name : label file path
 * @param labels : vector of string labels
 */
void readLabelsFile(const std::string& file_name, std::vector<std::string>& labels)
{
    std::ifstream file(file_name);
    if (!file) {
       return;
     }
    labels.clear();
     std::string line;
     while (std::getline(file, line)) {
         labels.push_back(line);
     }

}

#endif /* utils_h */
