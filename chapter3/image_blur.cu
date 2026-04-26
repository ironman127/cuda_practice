#include<stdio.h>
#include<malloc.h>
#include<math.h>

#define STB_IMAGE_IMPLEMENTATION    // 必须在 include 之前！
#include<stb_image.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION    // 必须在 include 之前！
#include<stb_image_write.h>

#define IMAGE_PATH "../data/cow.jpg"
#define OUT_IMAGE_PATH "../data/blured_cow.jpg"
#define pixel_t unsigned char
#define PATH_SIZE 20

__global__ void image_blur_kernal(pixel_t *img, pixel_t *blured_img, int width, int hight, int channels) {
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    
    int lower_row = row - PATH_SIZE >= 0 ? row - PATH_SIZE : 0;
    int upper_row = row + PATH_SIZE < hight ? row + PATH_SIZE : hight;
    int lower_col = col - PATH_SIZE >= 0 ? col - PATH_SIZE : 0;
    int upper_col = col + PATH_SIZE < width ? col + PATH_SIZE : width;
    
    if (row >= hight || col >= width) {
        return;
    }

    for (int c=0; c < channels; ++c) {
        int pixel_val_sum = 0;
        for (int r=lower_row; r < upper_row; ++r) {
            for (int cl=lower_col; cl < upper_col; ++cl) {
                int pixel_idx = r * width + cl;
                pixel_val_sum += img[pixel_idx*channels+c];
            }
        }

        int pixel_idx = row * width + col;
        blured_img[pixel_idx*channels+c] = pixel_val_sum / ((upper_col - lower_col) * (upper_row - lower_row));
    }
    
}

void image_blur(pixel_t *img_h, pixel_t *blured_img_h, int width, int hight, int channels) {
    pixel_t *img_d, *blured_img_d;
    int pixel_num = width * hight * channels;

    cudaMalloc((void**)&img_d, pixel_num);
    cudaMalloc((void**)&blured_img_d, pixel_num);

    cudaMemcpy(img_d, img_h, pixel_num, cudaMemcpyHostToDevice);

    dim3 block_dim = {16, 32, 1};
    dim3 grid_dim = {ceil(width / block_dim.x), ceil(hight / block_dim.y), 1};

    image_blur_kernal<<<grid_dim, block_dim>>>(img_d, blured_img_d, width, hight, channels);

    cudaMemcpy(blured_img_h, blured_img_d, pixel_num, cudaMemcpyDeviceToHost);

    cudaFree(img_d);
    cudaFree(blured_img_d);
}

int main() {
    int width, height, channels;
    pixel_t *img_h = stbi_load(IMAGE_PATH, &width, &height, &channels, 0);

    if (!img_h) { 
        printf("加载图片失败!\n"); 
        return -1; 
    }
    printf("图片尺寸: %d × %d, 通道数: %d\n", width, height, channels);

    pixel_t *blured_image_h = (pixel_t*) malloc(sizeof(pixel_t) * width * height * channels);
    image_blur(img_h, blured_image_h, width, height, channels);

    stbi_write_jpg(OUT_IMAGE_PATH, width, height, channels, blured_image_h, 90);
    printf("模糊结果已保存: output_blur.jpg\n");

    stbi_image_free(img_h);
    free(blured_image_h);

    return 0;
}