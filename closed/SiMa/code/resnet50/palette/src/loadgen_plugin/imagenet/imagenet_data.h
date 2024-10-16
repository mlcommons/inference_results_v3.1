#ifndef __GST_IMAGENET_H__
#define __GST_IMAGENET_H__

#include <iostream>
#include <cstdint>


template <int C, int D>
struct ImgNet {
    size_t num_images;
    size_t image_size;
    int8_t (*data)[C][D][D]; // [num_images][num_channels][image_dim][image_dim]
};

typedef ImgNet<3, 224> ImageNet;

ImageNet* load_8bit_imagenet(const char* filename, const size_t num_images);

#endif /* __GST_IMAGENET_H__ */
