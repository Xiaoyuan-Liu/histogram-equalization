%test histeq
I = imread('color.jpg');
[J] = Histogram_equalization(I);
%figure, imhist(J,25);%输出均衡化后的图像，其中灰度图是均匀分布的直方图均衡化，彩图是RGB转HSI对I通道进行直方图均衡化
figure, imshow(J)
[K]=histeq(I);%matlab的histeq函数进行均衡化，作为对比
%figure, imhist(K,25);
figure, imshow(I)
%figure, imhist(I);%输出原图像
figure, imshow(K)
%% 