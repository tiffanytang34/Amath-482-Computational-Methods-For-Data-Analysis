%% test 1
clc;clear all;close all;
load('cam1_1.mat')
load('cam2_1.mat')
load('cam3_1.mat')
%implay(vidFrames1_1)
numFrames11 = size(vidFrames1_1,4); 
numFrames21 = size(vidFrames2_1,4); 
numFrames31 = size(vidFrames3_1,4); 
%%

width = 50;
filter = zeros(480, 640);

filter(300-2.6*width:1:300+2.6*width, 350-width:1:350+width)=1;


data1 = [];
for j = 1: numFrames11
    X = vidFrames1_1(:,:,:,j);
    Xabw = rgb2gray(X);
    X2 = double(X);
    
    Xabw2 = double(Xabw);
    Xf = Xabw2.*filter;
    thresh = Xf > 250;
    indeces = find(thresh);
    [Y, X] = ind2sub(size(thresh), indeces);
    
    data1 = [data1; mean(X), mean(Y)];
    %imshow(uint8(Xf)); drawnow
end

filter = zeros(480, 640);

filter(100-width:1:350+width, 290-1.3*width:1:290+1.3*width)=1;


data2 = [];
for j = 1: numFrames21
    X = vidFrames2_1(:,:,:,j);
    Xabw = rgb2gray(X);
    X2 = double(X);
    
    Xabw2 = double(Xabw);
    Xf = Xabw2.*filter;
    thresh = Xf > 250;
    indeces = find(thresh);
    [Y, X] = ind2sub(size(thresh), indeces);
    
    data2 = [data2; mean(X), mean(Y)];
    
    %imshow(uint8(Xf)); drawnow
end

filter = zeros(480, 640);

filter(250-1*width:1:250+2*width, 360-2.6*width:1:360+2.6*width)=1;


data3 = [];
for j = 1: numFrames31
    X = vidFrames3_1(:,:,:,j);
    Xabw = rgb2gray(X);
    X2 = double(X);
    
    Xabw2 = double(Xabw);
    Xf = Xabw2.*filter;
    thresh = Xf > 248;
    indeces = find(thresh);
    [Y, X] = ind2sub(size(thresh), indeces);
    
    data3 = [data3; mean(X), mean(Y)];
    %imshow(uint8(Xf)); drawnow
end

[M, I] = min(data1(1:20, 2));
data1 = data1(I:end, :);
[M, I] = min(data2(1:20, 2));
data2 = data2(I:end, :);
[M, I] = min(data3(1:20, 2));
data3 = data3(I:end, :);

data2 = data2(1:length(data1), :);
data3 = data3(1:length(data1), :);

datasum = [data1';data2';data3']

[m,n] = size(datasum);
avg = mean(datasum, 2);
datasum = datasum - repmat(avg, 1, n);

[u,s,v] = svd(datasum'/sqrt(n-1))
lambda = diag(s).^2;
Y = datasum'* v; % principal components projection

sig = diag(s);

figure()
plot(1:6, lambda/sum(lambda), 'mo', 'Linewidth', 2)
title("Case 1: Energy for different POD modes ")
xlabel("modes"); ylabel("Energy ")

figure()
subplot(2, 1, 1)
plot(1:218, datasum(4,:), 1:218, datasum(6,:), 'Linewidth', 2)
ylabel("Displacement (pixels)"); xlabel("Time(frames)")
title("Case 1:original displacement across Z axis and XY- plane")
legend('Z', 'XY')
subplot(2, 1, 2)
plot(1:218, Y(:, 1), 'r', 'Linewidth', 2)
ylabel('Displacement (pixels)'); xlabel('Time (frames)')
title ('Case 1: Displacement across principal component directions')
legend('PC1')

%% tEST 2
clc;clear all;close all;
load('cam1_2.mat')
load('cam2_2.mat')
load('cam3_2.mat')
%implay(vidFrames1_1)
numFrames12 = size(vidFrames1_2,4); 
numFrames22 = size(vidFrames2_2,4); 
numFrames32 = size(vidFrames3_2,4);

width = 50;
filter = zeros(480, 640);

filter(300-2.6*width:1:300+2.6*width, 350-width:1:350+2*width)=1;


data1 = [];
for j = 1: numFrames12
    X = vidFrames1_2(:,:,:,j);
    Xabw = rgb2gray(X);
    X2 = double(X);
    
    Xabw2 = double(Xabw);
    Xf = Xabw2.*filter;
    thresh = Xf > 250;
    indeces = find(thresh);
    [Y, X] = ind2sub(size(thresh), indeces);
    
    data1 = [data1; mean(X), mean(Y)];
    
end


width = 50;
filter = zeros(480, 640);

filter(100-width:1:375+width, 215-width:1:290+2.7*width)=1;


data2 = [];
for j = 1: numFrames22
    X = vidFrames2_2(:,:,:,j);
    Xabw = rgb2gray(X);
    X2 = double(X);
    
    Xabw2 = double(Xabw);
    Xf = Xabw2.*filter;
    thresh = Xf > 249;
    indeces = find(thresh);
    [Y, X] = ind2sub(size(thresh), indeces);
    
    data2 = [data2; mean(X), mean(Y)];
  
end

width = 50;
filter = zeros(480, 640);

filter(250-1*width:1:250+2.7*width, 360-2.5*width:1:360+2.7*width)=1;


data3 = [];
for j = 1: numFrames32
    X = vidFrames3_2(:,:,:,j);
    Xabw = rgb2gray(X);
    X2 = double(X);
    
    Xabw2 = double(Xabw);
    Xf = Xabw2.*filter;
    thresh = Xf > 245;
    indeces = find(thresh);
    [Y, X] = ind2sub(size(thresh), indeces);
    
    data3 = [data3; mean(X), mean(Y)];
    
end

[M, I] = min(data1(1:20, 2));
data1 = data1(I:end, :);
[M, I] = min(data2(1:20, 2));
data2 = data2(I:end, :);
[M, I] = min(data3(1:20, 2));
data3 = data3(I:end, :);

data2 = data2(1:length(data1), :);
data3 = data3(1:length(data1), :);

datasum = [data1';data2';data3']

[m,n] = size(datasum);
avg = mean(datasum, 2);
datasum = datasum - repmat(avg, 1, n);

[u,s,v] = svd(datasum'/sqrt(n-1))
lambda = diag(s).^2;
Y = datasum'* v; % principal components projection

sig = diag(s);

figure()
plot(1:6, lambda/sum(lambda), 'mo', 'Linewidth', 2)
title("Case 2: Energy for different POD modes ")
xlabel("POD modes"); ylabel("Energy ")

figure()
subplot(2, 1, 1)
plot(1:295, datasum(2,:), 1:295, datasum(1,:), 'Linewidth', 2)
ylabel("Displacement (pixels)"); xlabel("Time(frames)")
title("Case 2:original displacement across Z axis and XY- plane(cam 1)")
legend('Z', 'XY')
subplot(2, 1, 2)
plot(1:295, Y(:, 1), 1:295, Y(:,2),'r', 'Linewidth', 2)
ylabel('Displacement (pixels)'); xlabel('Time (frames)')
title ('Case 2: Displacement across principal component directions')
legend('PC1', 'PC2')

%% tEST 3
clc;clear all;close all;
load('cam1_3.mat')
load('cam2_3.mat')
load('cam3_3.mat')
%implay(vidFrames1_1)
numFrames13 = size(vidFrames1_3,4); 
numFrames23 = size(vidFrames2_3,4); 
numFrames33 = size(vidFrames3_3,4);

width = 50;
filter = zeros(480, 640);

filter(275-width:1:300+3*width, 350-1.5*width:1:350+2*width)=1;


data1 = [];
for j = 1: numFrames13
    X = vidFrames1_3(:,:,:,j);
    Xabw = rgb2gray(X);
    X2 = double(X);
    
    Xabw2 = double(Xabw);
    Xf = Xabw2.*filter;
    thresh = Xf > 250;
    indeces = find(thresh);
    [Y, X] = ind2sub(size(thresh), indeces);
    
    data1 = [data1; mean(X), mean(Y)];
    
end


width = 50;
filter = zeros(480, 640);

filter(100-width:1:375+width, 215-width:1:290+2.7*width)=1;


data2 = [];
for j = 1: numFrames23
    X = vidFrames2_3(:,:,:,j);
    Xabw = rgb2gray(X);
    X2 = double(X);
    
    Xabw2 = double(Xabw);
    Xf = Xabw2.*filter;
    thresh = Xf > 249;
    indeces = find(thresh);
    [Y, X] = ind2sub(size(thresh), indeces);
    
    data2 = [data2; mean(X), mean(Y)];
  
end

width = 50;
filter = zeros(480, 640);

filter(250-1.8*width:1:250+2.3*width, 360-2.5*width:1:360+2.7*width)=1;


data3 = [];
for j = 1: numFrames33
    X = vidFrames3_3(:,:,:,j);
    Xabw = rgb2gray(X);
    X2 = double(X);
    
    Xabw2 = double(Xabw);
    Xf = Xabw2.*filter;
    thresh = Xf > 245;
    indeces = find(thresh);
    [Y, X] = ind2sub(size(thresh), indeces);
    
    data3 = [data3; mean(X), mean(Y)];    
end

[M, I] = min(data1(1:20, 2));
data1 = data1(I:end, :);
[M, I] = min(data2(1:20, 2));
data2 = data2(I:end, :);
[M, I] = min(data3(1:20, 2));
data3 = data3(I:end, :);

data2 = data2(1:length(data3), :);
data1 = data1(1:length(data3), :);

datasum = [data1';data2';data3']

[m,n] = size(datasum);
avg = mean(datasum, 2);
datasum = datasum - repmat(avg, 1, n);

[u,s,v] = svd(datasum'/sqrt(n-1))
lambda = diag(s).^2;
Y = datasum'* v; % principal components projection

figure()
plot(1:6, lambda/sum(lambda), 'mo', 'Linewidth', 2)
title("Case 3: Energy for different POD modes ")
xlabel("POD modes"); ylabel("Energy ")

figure()
subplot(2, 1, 1)
plot(1:236, datasum(1, :), 1:236, datasum(2,:),'Linewidth', 2)
ylabel("Displacement (pixels)"); xlabel("Time (frames)")
title("Case 3:original displacement across Z axis and XY- plane（cam 1）")
legend('Z', 'XY')
subplot(2, 1, 2)
plot(1:236, Y(:, 1), 1:236, Y(:,2),1:236, Y(:, 3),1:236, Y(:, 4), 'r', 'Linewidth', 2)
ylabel('Displacement (pixels)'); xlabel('Time (frames)')
title ('Case 3: Displacement across principal component directions')
legend('PC1', 'PC2','PC3', 'PC4')

%% test 4
clc;clear all;close all;
load('cam1_4.mat')
load('cam2_4.mat')
load('cam3_4.mat')
%implay(vidFrames1_1)
numFrames14 = size(vidFrames1_4,4); 
numFrames24 = size(vidFrames2_4,4); 
numFrames34 = size(vidFrames3_4,4);

width = 50;
filter = zeros(480, 640);

filter(275-width:1:400+width, 350-1.5*width:1:370+2*width)=1;


data1 = [];
for j = 1: numFrames14
    X = vidFrames1_4(:,:,:,j);
    Xabw = rgb2gray(X);
    X2 = double(X);
    
    Xabw2 = double(Xabw);
    Xf = Xabw2.*filter;
    thresh = Xf > 247;
    indeces = find(thresh);
    [Y, X] = ind2sub(size(thresh), indeces);
    
    data1 = [data1; mean(X), mean(Y)];
    
end


width = 50;
filter = zeros(480, 640);

filter(100-width:1:350+width, 215-width:1:290+2.7*width)=1;


data2 = [];
for j = 1: numFrames24
    X = vidFrames2_4(:,:,:,j);
    Xabw = rgb2gray(X);
    X2 = double(X);
    
    Xabw2 = double(Xabw);
    Xf = Xabw2.*filter;
    thresh = Xf > 249;
    indeces = find(thresh);
    [Y, X] = ind2sub(size(thresh), indeces);
    
    data2 = [data2; mean(X), mean(Y)];
  
end

width = 50;
filter = zeros(480, 640);

filter(150-width:1:250+1*width, 360-1.8*width:1:360+2.9*width)=1;


data3 = [];
for j = 1: numFrames34
    X = vidFrames3_4(:,:,:,j);
    Xabw = rgb2gray(X);
    X2 = double(X);
    
    Xabw2 = double(Xabw);
    Xf = Xabw2.*filter;
    thresh = Xf > 234;
    indeces = find(thresh);
    [Y, X] = ind2sub(size(thresh), indeces);
    
    data3 = [data3; mean(X), mean(Y)];    
end

[M, I] = min(data1(1:20, 2));
data1 = data1(I:end, :);
[M, I] = min(data2(1:20, 2));
data2 = data2(I:end, :);
[M, I] = min(data3(1:20, 2));
data3 = data3(I:end, :);

data2 = data2(1:length(data3), :);
data1 = data1(1:length(data3), :);

datasum = [data1';data2';data3']

[m,n] = size(datasum);
avg = mean(datasum, 2);
datasum = datasum - repmat(avg, 1, n);

[u,s,v] = svd(datasum'/sqrt(n-1))
lambda = diag(s).^2;
Y = datasum'* v; % principal components projection
sig = diag(s);
figure()
plot(1:6, lambda/sum(lambda), 'mo', 'Linewidth', 2)
title("Case 4: Energy for each POD mode ")
xlabel("POD modes"); ylabel("Energy ")

figure()
subplot(2, 1, 1)
plot(1:375, datasum(2,:),1:375, datasum(1,:),'Linewidth', 2)
ylabel("Displacement (pixels)"); xlabel("Time (frames)")
title("Case 4:original displacement across Z axis and XY- plane（cam 1）")
legend('Z', 'XY')
subplot(2, 1, 2)
plot(1:375, Y(:, 1), 1:375, Y(:,2),1:375, Y(:, 3), 'Linewidth', 2)
ylabel('Displacement (pixels)'); xlabel('Time (frames)')
title ('Case 4: Displacement across principal component directions')
legend('PC1', 'PC2','PC3')