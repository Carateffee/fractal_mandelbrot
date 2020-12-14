#include <stdint.h>

class Bmp256 {										//自定义图像类
#pragma pack(2) 									// 设定变量以n = 2字节对齐方式
	struct Header {									// 头信息
		uint16_t bfType = 0x4D42;
		uint32_t bfSize;
		uint16_t bfReserved1 = 0;
		uint16_t bfReserved2 = 0;
		uint32_t bfOffBits = 54 + 256 * 4;
		uint32_t biSize = 40;
		int32_t  biWidth;
		int32_t  biHeight;
		uint16_t biPlanes = 1;
		uint16_t biBitCount = 8;
		uint32_t biCompression = 0;
		uint32_t biSizeImage = 0;
		int32_t  biXPelsPerMeter = 0;
		int32_t  biYPelsPerMeter = 0;
		uint32_t biClrUsed = 256;
		uint32_t biClrImportant = 256;
	} header;
#pragma pack()										// 默认值8
	int32_t rowSize;								// 行大小
	struct {										// 像素点通道结构
		uint8_t B, G, R, A;
		void set(uint8_t r, uint8_t g, uint8_t b) { R = r; G = g; B = b; A = 0; }	// 设置颜色函数
	} palette[256];									// 256个颜色的调色板
	uint8_t *buffer = nullptr;						// 图像缓存

	void calc_palette();

public:
	Bmp256(int width, int height);					// 类构造函数
	~Bmp256() { delete[] buffer; }					// 类析构函数
	int width()  const { return header.biWidth; }	// 获取图像宽度
	int height() const { return header.biHeight; }	// 获取图像高度
	uint8_t& operator()(int row, int col) { return buffer[row * rowSize + col]; }	// get/set the pixel
	void save(const char* file);					// 保存图像
};

Bmp256::Bmp256(int width, int height) {
	header.biWidth = width;							// 从头信息中获取宽度和高度
	header.biHeight = height;
	rowSize = ((width + 3) / 4) * 4;				// 计算行大小
	int buffSize = rowSize * height;				// 图像整体缓存的大小
	header.bfSize = header.bfOffBits + buffSize;	
	calc_palette();									// 初始化调色板颜色
	buffer = new uint8_t[buffSize];					// 新建图像缓存
}

void Bmp256::calc_palette() {
	for (int i = 0; i < 64; ++i) {
		palette[i].set(255, 255 - i * 4, 0);
		palette[i + 64].set(255 - i * 2, 0, i * 2);
		palette[i + 128].set(127 - i * 2, 0, 128 + i * 2);
		palette[i + 192].set(0, 0, 255 - i * 3);
	}
	palette[0].set(0, 0, 0);
}

#include <iostream>
#include <fstream>
void Bmp256::save(const char* file_name) {			// 保存
	std::ofstream of(file_name, std::ios::binary);
	of.write((char *)&header, 54);
	of.write((char *)palette, 256 * 4);
	char* p = (char *)buffer;
	for (int i = 0; i < header.biHeight; ++i) {
		of.write(p, rowSize);
		p += rowSize;
	}
}


#include <complex>
using complex = std::complex<double>;				//复数类

const double RMIN = -2, RMAX = 1, IMIN = -1, IMAX = 1;// 实部和虚部的范围
const int W = 12 * 1024;							// 宽度：12*1024
const double RESN = W / (RMAX - RMIN);				// 实部单位像素数12*1024/（1-(-2)）=4*1024
const int H = (IMAX - IMIN) * RESN;					// 高度：（1-(-1)）*4*1024=8*1024
const int MI = 1;

int Mandelbrot(complex c) {							// 曼德博集合是一种在复平面上组成分形的点的集合
	complex z;
	for (int k = 256 * MI - 1; k >= 0; --k) {
		z = z * z + c;
		if (std::norm(z) > 4) return k / MI;		//计算分形
	}
	return 0;
}

#include <chrono>
#include <omp.h>
using namespace std::chrono;

int main() {
	Bmp256 bmp{W, H};

	auto t1 = steady_clock::now();					// 单调计时时钟
	omp_set_num_threads(8);
	#pragma omp parallel for schedule(dynamic) 
	for (int j = 0; j < H; ++j){
	for (int i = 0; i < W; ++i)
		bmp(j, i) = Mandelbrot(complex{RMIN + i / RESN, IMIN + j / RESN});//{实部，虚部}
	}
	auto t2 = steady_clock::now();

	bmp.save("Mandelbrot12k.bmp");
	std::cout << "OMP dynamic run time: " << duration_cast<duration<double>>(t2 - t1).count() << " seconds.\n";
}
