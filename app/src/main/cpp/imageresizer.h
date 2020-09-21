//
// Created by rohith on 7/24/18.
//

#ifndef TFLITENATIVE_IMAGERESIZER_H
#define TFLITENATIVE_IMAGERESIZER_H

#include "logs.h"

#include <cstdint>


/**
 *center crop and resize the image
 *
 */
void cropResizeImage(unsigned char *inputpixel, int w_image, int h_image ,int ch_image, uint8_t* output, int w_network, int h_network)
{
    LOGV("in width %d, in height %d",w_image, h_image);

    if(h_image >= w_image)//portrait mode
    {
        LOGE("portrait");
        const int skipheight = (h_image - w_image) / 2;
        LOGE("skipheight %d", skipheight);

        int croped_width = w_image;
        int croped_height = w_image;

        unsigned char* start_address = inputpixel + (skipheight*croped_width)*ch_image;
        uint8_t* rgb_in = output;

        for(int y =0;y<h_network; y++)
        {

            const int in_y = y * (croped_height) / h_network; // 0 = 0x1356/224, 6 = 1x1356/224 ,12 = 2x1356/224, 18 = 3x1356/224.....1349 = 113x1356/224 [total height = 2392 ]


            unsigned char* in_row = start_address + in_y * w_image*ch_image;
            uint8_t* out_row =  rgb_in+ (y*w_network*3); //ou_input + (0*224*3), (1*224*3) ,(2*224*3), (3*224*3) ....(223*224*3)

            for(int x=0;x<w_network;x++)
            {

                const int in_x = (x * croped_width) / w_network; //  0 = 0x1356/224, 6 = 1x1356/224 ,12 = 2x1356/224, 18 = 3x1356/224.....1349 = 113x1356/224 [total width = 1356 ]


                unsigned char* inpixel = in_row +in_x*ch_image;

                uint8_t* out_pixel =  out_row +(x*3);//out_row +(0x3), (1x3) ,(2x3), (3x3), (4x3)...(223x3)

                out_pixel[0] = inpixel[0];
                out_pixel[1] = inpixel[1];
                out_pixel[2] = inpixel[2];

            }
        }

    }
    else
    {
        LOGE("Landscape");

        const int skipwidth = (w_image - h_image) / 2;
        LOGE("skipheight %d", skipwidth);

        int croped_width = h_image;
        int croped_height = h_image;


        unsigned char* start_address = inputpixel;

        uint8_t* rgb_in = output;
        for(int y =0;y<h_network; y++)
        {

            const int in_y = y * (h_image) / h_network;

            unsigned char* in_row = start_address + in_y*w_image*ch_image;
            uint8_t* out_row =  rgb_in+ (y*w_network*3); //ou_input + (0*224*3), (1*224*3) ,(2*224*3), (3*224*3) ....(223*224*3)

            for(int x=0;x<w_network;x++)
            {


                const int in_x = (x * h_image) / w_network;

                unsigned char* inpixel = in_row +in_x*ch_image+skipwidth*ch_image;

                uint8_t* out_pixel =  out_row +(x*3);//out_row +(0x3), (1x3) ,(2x3), (3x3), (4x3)...(223x3)


                out_pixel[0] = inpixel[0];
                out_pixel[1] = inpixel[1];
                out_pixel[2] = inpixel[2];

            }
        }



    }
}

#endif //TFLITENATIVE_IMAGERESIZER_H
