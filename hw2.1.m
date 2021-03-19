%% 
clear all;clc;
[y, Fs] = audioread('/Users/tiffany/Dropbox/My Mac (Tiffany的MacBook Pro)/Desktop/UW/WI21/AMATH482/HW2/GNR.m4a');
trgnr = length(y)/Fs; % record time in seconds4plot((1:length(y))/Fs,y);

s=y'; % signal
L = trgnr;
n=length(s); % length of signal
t2 = linspace(0, L, n+1);
t=t2(1:n);
k =(1/L)*[0:n/2-1 -n/2:-1]; % 1/L instead of 2*pi/L
ks=fftshift(k); 
%%

%zs_fft = fft(s);
%s_filter = s_fft.*fftshift(60<abs(t)<250);
%s_bass = ifft(s_filter);

tau = 0:0.5:L;
a= 10;
sgt_spec = []
for i = 1:length(tau)
    g = exp(-a*(t-tau(i))).^2; % window function
    sg = g.* s;
    sgt = fft(sg);
    sgt_spec(:, i) = fftshift(abs(sgt));
end

pcolor(tau, ks, sgt_spec)
shading interp
ylim([0 400])
%set(gca, 'ylim', [0 500],'FrontSize', 14)
xlabel('time (t)'), ylabel('frequency (k)')
xlabel('time (t)')
ylabel('frequency (k)')
colorman(hot)