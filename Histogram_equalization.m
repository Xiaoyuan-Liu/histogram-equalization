function [output] = Histogram_equalization(input_image)
%first test the image is a RGB or gray image
    if numel(size(input_image)) == 3
    %this is a RGB image
    %here is just one method, if you have other ways to do the
    %equalization, you can change the following code
    r=input_image(:,:,1);
    v=input_image(:,:,2);
    b=input_image(:,:,3);
    r1 = hist_equal(r);
    v1 = hist_equal(v);
    b1 = hist_equal(b);
    figure,imshow(cat(3,r1,v1,b1));%对三个颜色通道分别进行均衡化
    %yuvImage = rgb2ycbcr(input_image);
    %y=yuvImage(:,:,1);%Y
    %u=yuvImage(:,:,2);%U
    %v=yuvImage(:,:,3);%V
    %y1=hist_equal2(y);
    figure,imshow(hist_equalByYuvImage(input_image));%采用RGB转YUV进行对Y通道进行均衡化    
    output=hist_equalByHsiImage(input_image);%采用RGB转HSI对I通道进行均衡化
    else
    %this is a gray image
    figure,imshow(hist_equal2(input_image));%概率分布函数为alnx的直方图均衡化
    figure,imshow(hist_equal3(input_image));%概率分布函数为ae^x的直方图均衡化
    [output] = hist_equal(input_image);
    end
end
function [yuvOutput] = hist_equalByYuvImage(input_image)
%采用RGB转YUV对彩图进行均衡化
    yuvImage = rgb2ycbcr(input_image);
    y=yuvImage(:,:,1);%Y
    u=yuvImage(:,:,2);%U
    v=yuvImage(:,:,3);%V
    y1=hist_equal(y);
    yuvOutput = ycbcr2rgb(cat(3,y1,u,v));    
end
function [hsiOutput]=hist_equalByHsiImage(input_image)
%采用RGB转HSI对彩图进行均衡化
    hsiImage = uint8(255*rgb2hsi(input_image));
    h=hsiImage(:,:,1);%H
    s=hsiImage(:,:,2);%S
    i=hsiImage(:,:,3);%I
    i1=hist_equal(i);
    hsiOutput=hsi2rgb(im2double(cat(3,h,s,i1)));
end
function [hsi] = rgb2hsi(rgb)
%RGB转HSI
    rgb = im2double(rgb);
    r = rgb(:, :, 1);
    g = rgb(:, :, 2);
    b = rgb(:, :, 3);
    
    num = 0.5 * ((r - g) + (r - b));
    den = sqrt((r - g).^2+(r - b).*(g - b));
    theta = acos(num./(den+eps));
    
    H=theta;
    H(b > g) = 2 * pi - H(b > g);
    H = H/(2 * pi);
    
    num = min(min(r , g), b);
    den = r + g + b;
    den(den == 0) = eps;
    S = 1 - 3.* num./den;
    H(S == 0) =0;
    I=(r + g + b)/3;
    
    hsi=cat(3, H, S, I);
end

function [rgb]=hsi2rgb(hsi)
%HSI转RGB
    H=hsi(:, :, 1) * 2 * pi;
    S=hsi(:, :, 2);
    I=hsi(:, :, 3);
    
    R=zeros(size(hsi, 1),size(hsi, 2));
    G=zeros(size(hsi, 1),size(hsi, 2));
    B=zeros(size(hsi, 1),size(hsi, 2));
    
    idx=find((0<=H)&(H<2*pi/3));
    B(idx)=I(idx) .* (1-S(idx));
    R(idx)=I(idx) .* (1+S(idx) .* cos(H(idx))./...
        cos(pi/3 - H(idx)));
    G(idx)=3*I(idx)-(R(idx)+B(idx));
    
    idx = find((2*pi/3 <= H)&(H < 4*pi/3));
    R(idx)=I(idx) .* (1-S(idx));
    G(idx)=I(idx) .* (1+S(idx) .* cos(H(idx)-2*pi/3) ./ ...
        cos(pi-H(idx)));
    B(idx)=3*I(idx)-(R(idx)+G(idx));
    
    idx=find((4*pi/3 <=H)&(H<=2*pi));
    G(idx)=I(idx) .* (1-S(idx));
    B(idx)=I(idx) .* (1+S(idx) .* cos(H(idx)-4*pi/3) ./ ...
        cos(5*pi/3-H(idx)));
    R(idx)=3*I(idx)-(G(idx)+B(idx));
    
    rgb=cat(3,R,G,B);
    rgb=max(min(rgb,1),0);
end
function [output4] = hist_equal3(input_channel)
%概率分布函数为ae^x的直方图均衡化    
    [r,c]=size(input_channel);%size可以获取矩阵的行列数
    numOfLevel = zeros(1,256);%创建于一个1*256的矩阵，即数组
    for i=1:r%注意，matlab数组索引从1开始
        for j=1:c
           numOfLevel(1,input_channel(i,j)+1) = numOfLevel(1,input_channel(i,j)+1)+1;%像素是从0~255，所以像素值+1才是像素在数组中的索引
        end
    end
    
    func=zeros(1,256);
    sumOfLevel = 0;
    for i=1:256
        sumOfLevel=sumOfLevel+numOfLevel(1,i);
        %func(1,i)=((256)/(r*c))*sumOfLevel;%像素从0~255
        func(1,i)=uint8(log((exp(256)-1)/(r*c)*sumOfLevel-1));
    end
    imOutput=zeros(r,c);
    for i=1:r
        for j=1:c
            imOutput(i,j)=func(1,(input_channel(i,j)+1));
        end
    end
    %imOutput(1,1)=imOutput(1,1)+1;
    output4=uint8(imOutput);
end
function [output3] = hist_equal2(input_channel)
%you should complete this sub-function
%概率分布函数为alnx的直方图均衡化
    [r,c]=size(input_channel);%size可以获取矩阵的行列数
    numOfLevel = zeros(1,256);%创建于一个1*256的矩阵，即数组
    for i=1:r%注意，matlab数组索引从1开始
        for j=1:c
           numOfLevel(1,input_channel(i,j)+1) = numOfLevel(1,input_channel(i,j)+1)+1;%像素是从0~255，所以像素值+1才是像素在数组中的索引
        end
    end
    
    func=zeros(1,256);
    sumOfLevel = 0;
    for i=1:256
        sumOfLevel=sumOfLevel+numOfLevel(1,i);
        %func(1,i)=((256)/(r*c))*sumOfLevel;%像素从0~255
        syms x
        func(1,i)=uint8(solve((x+1)*log(x+1)==257*log(257)/(r*c)*sumOfLevel,x));
    end
    imOutput=zeros(r,c);
    for i=1:r
        for j=1:c
            imOutput(i,j)=func(1,(input_channel(i,j)+1));
        end
    end
    %imOutput(1,1)=imOutput(1,1)+1;
    output3=uint8(imOutput);
end
function [output2] = hist_equal(input_channel)
    %均匀分布的直方图均衡化
    [r,c]=size(input_channel);%size可以获取矩阵的行列数
    numOfLevel = zeros(1,256);%创建于一个1*256的矩阵，即数组
    for i=1:r%注意，matlab数组索引从1开始
        for j=1:c
           numOfLevel(1,input_channel(i,j)+1) = numOfLevel(1,input_channel(i,j)+1)+1;%像素是从0~255，所以像素值+1才是像素在数组中的索引
        end
    end
    
%    probabilityOfLevel = zeros(1,256);%归一化处理。一定需要吗？
%    for i=1:256
%        probabilityOfLevel(1,i)=numOfLevel(1,i)/(r*c);
%    end

    func=zeros(1,256);
    sumOfLevel = 0;
    for i=1:256
        sumOfLevel=sumOfLevel+numOfLevel(1,i);
        func(1,i)=(256/(r*c))*sumOfLevel;%像素从0~255
    end
    imOutput=zeros(r,c);
    for i=1:r
        for j=1:c
            imOutput(i,j)=func(1,(input_channel(i,j)+1));
        end
    end
    %imOutput(1,1)=imOutput(1,1)+1;
    output2=uint8(imOutput);
end