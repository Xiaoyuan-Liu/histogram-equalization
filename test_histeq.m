%test histeq
I = imread('color.jpg');
[J] = Histogram_equalization(I);
%figure, imhist(J,25);%������⻯���ͼ�����лҶ�ͼ�Ǿ��ȷֲ���ֱ��ͼ���⻯����ͼ��RGBתHSI��Iͨ������ֱ��ͼ���⻯
figure, imshow(J)
[K]=histeq(I);%matlab��histeq�������о��⻯����Ϊ�Ա�
%figure, imhist(K,25);
figure, imshow(I)
%figure, imhist(I);%���ԭͼ��
figure, imshow(K)
%% 