%% Clean workspace
clear all; close all; clc

load('subdata.mat') % Imports the data as the 262144x49 (space by time) matrix called subdata

L = 10; % spatial domain
n = 64; % Fourier modes
x2 = linspace(-L,L,n+1); x = x2(1:n); y =x; z = x;
k = (2*pi/(2*L))*[0:(n/2 - 1) -n/2:-1]; ks = fftshift(k);

[X,Y,Z]=meshgrid(x,y,z);
[Kx,Ky,Kz]=meshgrid(ks,ks,ks); % grid in frequency domain

%for j=1:49
%    Un(:,:,:)=reshape(subdata(:,j),n,n,n);
%    M = max(abs(Un),[],'all');
%    close all, isosurface(X,Y,Z,abs(Un)/M,0.7)
%    axis([-20 20 -20 20 -20 20]), grid on, drawnow
%    pause(1)
%
%% problem 1 - center frequency
Utave = zeros(64, 64, 64)

for j = 1:49
   Un(:,:,:)=reshape(subdata(:,j),n,n,n);
   Utn = fftshift(fftn(Un));
   Utave = Utave + Utn;
end
Utave = abs(Utave./49);
[maxVals, indices1] = max(Utave(:));
[kx0, ky0, kz0] = ind2sub([n,n,n], indices1);
Kx0 = Kx(kx0, ky0, kz0)
Ky0 = Ky(kx0, ky0, kz0)
Kz0 = Kz(kx0, ky0, kz0)
%figure(1)
%isosurface(Kx,Ky,Kz,abs(Utave)/M, 0.7)

%% problem 2 - filter
locations = zeros(3, 49);
tau = 0.5;
filter = exp(-tau*((Kx-Kx0).^2+ (Ky-Ky0).^2+ (Kz-Kz0).^2)); % Define the filter
for j = 1:49
    Un(:,:,:)=reshape(subdata(:,j),n,n,n);
    Utn = fftshift(fftn(Un));
    Unft = filter.*Utn; % Apply the filter to the signal in frequency space
    Unf = ifftn(Unft); % the signal in time space
    
    M = max(abs(Unf),[],'all');
    indices2 = find(abs(Unf)==M);
    [a, b, c] = ind2sub([n, n, n], indices2);
    locations(1, j) = X(a,b,c);
    locations(2, j) = Y(a,b,c);
    locations(3, j) = Z(a,b,c);
end
figure(1)
plot3(locations(1, :), locations(2, :), locations(3,:))
title('Submarine Movement trajectory', 'Fontsize', 15)
xlabel('x')
ylabel('y')
zlabel('z')

x49 = locations(1, 49);
y49 = locations(2, 49);
z49 = locations(3, 49);
L = sprintf('The 49th location of the submarine is at %s %d %f.', x49, y49, z49);

%set(gca,'Fontsize',16,'Xlim',[-28 28])
%xlabel('frequency (k)'), ylabel('realizations'), zlabel('|fft(u)|')
%% Problem 3 - get the table of the 49 2-D positions in each time point
num = [1:49]
l = [locations(1,:);locations(2,:);locations(3, :)]
XY_locations = table([num;l])
writetable(XY_locations, 'xy1.csv')
type 'xy1.csv'