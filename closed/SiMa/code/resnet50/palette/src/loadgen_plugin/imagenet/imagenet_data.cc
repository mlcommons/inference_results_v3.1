
#include <iostream>
#include <cstdint>

#include "imagenet_data.h"



template <int C, int D>
ImgNet<C, D>* load_8bit_images(const char* filename, const size_t num_images) {
    printf("Loading file: %s\n", filename);
    FILE *fh = fopen(filename, "r");
    const size_t total_elements = num_images * C * D * D;
    int8_t (*data)[C][D][D] = new int8_t[num_images][C][D][D];
    printf("Loading %ld bytes...\n", total_elements);
    size_t bytes_read = fread(data, sizeof(int8_t), total_elements, fh);
    fclose(fh);
    ImgNet<C, D> *imgnet;
    imgnet = new ImgNet<C, D>();
    imgnet->num_images = num_images;
    imgnet->data = data;
    imgnet->image_size = C * D * D;
    printf("Done Loading.\n");
    return imgnet;
}

ImageNet* load_8bit_imagenet(const char* filename, const size_t num_images) {
    return load_8bit_images<3, 224>(filename, num_images);
}


// int main(int argc, char *argv[]) {
//     const size_t num_images = 50000;
//     const size_t num_channels = 3;
//     const size_t image_dim = 224;
//     const size_t total_elements = num_images * num_channels * image_dim * image_dim;
//     printf("Imagenext [%ld][%ld][%ld][%ld]\n", num_images, num_channels, image_dim, image_dim);
// 
//     ImageNet* imagenet = load_8bit_imagenet(argv[1], num_images);
// 
//     printf("Loaded file.\n");
//     printf("Checking Values...\n");
//     printf("imagenet[0][0][0][0] = %d\n", imagenet->data[0][0][0][0]);
//     printf("imagenet[1][0][0][0] = %d\n", imagenet->data[1][0][0][0]);
//     printf("imagenet[0][1][0][0] = %d\n", imagenet->data[0][1][0][0]);
//     printf("imagenet[0][0][1][0] = %d\n", imagenet->data[0][0][1][0]);
//     printf("imagenet[0][0][0][1] = %d\n", imagenet->data[0][0][0][1]);
//     printf("imagenet[2][2][2][2] = %d\n", imagenet->data[2][2][2][2]);
// 
//     /*  Expected output:
// Spot Checks: imgs[0][0][0][0] 56
// Spot Checks: imgs[1][0][0][0] 51
// Spot Checks: imgs[0][1][0][0] 68
// Spot Checks: imgs[0][0][1][0] 22
// Spot Checks: imgs[0][0][0][1] 50
// Spot Checks: imgs[2][2][2][2] -24
//     */
// 
//     return 0;
// }
// 
