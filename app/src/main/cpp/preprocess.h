//
//  preprocess.h
//  onnxRuntimeInference
//
//  Created by rohith kvsp on 9/13/20.
//  Copyright © 2020 rohith kvsp. All rights reserved.
//

#ifndef preprocess_h
#define preprocess_h

/**
 * Method to convert HWC to CHW format
 * @param input : normalized image pointer with HWC format
 * @param height : height of image
 * @param width : width of image
 * @param channels : channels of image
 * @param output : normalized image pointer with CHW format
 */
void HWCtoCHW(float* input,int height, int width, int channels, float* output)
{   /// image[i,j,c] = img[(width*i + j )*channels +c] - > HWC : image[c,i,j] = img[(height*c +i)*width + j] - > CHW
    for(int c = 0; c < channels; c++)
    {
        for(int i = 0; i<height; i++ )
        {
            for(int j=0; j<width; j++)
            {
                output[ (height*c +i)*width + j ] = input[ (width*i + j )*channels +c];
            }
        }
    }

    return;

}

/**
 *  method to skip Alpha channel and normalize the image
 * @param input : RGBA img pointer from bitmap
 * @param height : height of img
 * @param width : width of img
 * @param channels : channels of img (4 RBA)
 * @param output : normalized imag pointer
 * @param mean : meean values
 * @param std : std values;
 */

void preprocess(uint8_t* input, int height, int width, int channels, float* output, std::vector<float> mean, std::vector<float> std)
{

    int channel_out = 3;
    for(int i=0; i<height; i++)
    {
        for(int j=0; j<width; j++)
        {
            for(int c=0;c <channel_out; c++) //skip alpha channae;
            {
                float normalize = input[(i*width +j)*channels + c] / 255.0f;
                output[(i*width +j)*channel_out + c] = ( normalize - mean[c])/std[c];

            }
        }
    }
    
    return;
    
}

#endif /* preprocess_h */
