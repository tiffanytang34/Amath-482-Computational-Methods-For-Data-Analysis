clear all; close all;clc;
%% ski_drop
vid1 = VideoReader('ski_drop_low.mp4');
dt = 1/vid1.Framerate;
t = 0:dt:vid1.Duration;
vidFrames = read(vid1);
numFrames = get(vid1, 'NumFrames');

for i = 1:numFrames
    v_mat2 = rgb2gray(vidFrames(:, :, :, i));
    v_reshape = reshape(v_mat2, [], 1);
    X(:, i) = double(v_reshape);
end

%% DMD
X1 = X(:, 1:end-1);
X2 = X(:, 2:end);

% SVD of X1 and Computation of ~S

[U, Sigma, V] = svd(X1,'econ');
r = 2;
U_r = U(:, 1:r);
Sigma_r = Sigma(1:r, 1:r);
V_r = V(:, 1:r);
S_r = U_r'*X2*V_r*diag(1./diag(Sigma_r));
[eV_r, D] = eig(S_r); % compute eigenvalues + eigenvectors
mu = diag(D); % extract eigenvalues
omega = log(mu)/dt; % continuous-time eval
Phi = X2*V_r/Sigma_r*eV_r;

%% singular value
figure(1)
subplot(2,1,1)
plot(diag(Sigma),'ko','Linewidth',2)
title('Singular Value Spectrum')
set(gca,'Fontsize',16,'Xlim',[0 500])
xlabel('modes')
ylabel('singluar values')
subplot(2,1,2)
semilogy(diag(Sigma),'ko','Linewidth',2)
set(gca,'Fontsize',16,'Xlim',[0 500])
xlabel('modes')
ylabel('singluar values in log')
%%  Plotting Eigenvalues (omega)
figure(2)
line = -10:10;

plot(zeros(length(line),1),line,'k','Linewidth',2) % imaginary axis
hold on
plot(line,zeros(length(line),1),'k','Linewidth',2) % real axis
plot(real(omega)*dt,imag(omega)*dt,'r.','Markersize',15)
xlabel('Re(\omega)')
ylabel('Im(\omega)')
title('\omega')
set(gca,'FontSize',16,'Xlim',[-0.1 0.1],'Ylim',[-0.1 0.1])
%% DMD solution
b = Phi\X1(:,1);
u_modes = zeros(r, length(t));
t = (0:length(t)-1)*dt;
for i = 1:length(t)
    u_modes(:, i) = b.*exp(omega*t(i));
end
Xdmd = Phi * u_modes; 
%% sparse DMD
Xsparse = X1- abs(Xdmd);

R = Xsparse.* (Xsparse<0);
X_lr = R+abs(Xdmd); % low dimension DMD with R
% makeit look better on imshow
X_sp = Xsparse-R-R; % Xsparse without R

X_sp1 = (Xsparse-min(Xsparse))./(max(Xsparse)- min(Xsparse));
%% display 

v_back = uint8(reshape(Xdmd, 540, 960, 453));
v_fore = uint8(reshape(Xsparse, 540, 960, 453));
v_back_r = uint8(reshape(X_lr, 540, 960, 453));
v_fore_rreduction = uint8(reshape(X_sp, 540, 960, 453));
v_fore_nor = uint8(reshape(X_sp1, 540, 960, 453));
%% figure
figure(3)
subplot(2, 2, 1), 
imshow(v_back(:, :, 200))

title('Background Video')
subplot(2, 2, 2), 
imshow(v_fore(:, :, 200))
title('Foreground Video')
subplot(2, 2, 3), 
imshow(v_back_r(:, :, 200))
title('Background Video with R')
subplot(2, 2, 4), 
imshow(v_fore_rreduction(:, :, 200))
title('Foreground Video with 2R Subtraction ')
%% visualize normalized sparse matrix
figure (4)
sample = reshape(X_sp1(:, 200), 540, 960);
imshow(sample)
title('Foreground Video with Normalization')