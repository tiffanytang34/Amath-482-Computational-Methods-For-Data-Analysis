clc; clear all;close all;
% load train data
[images, labels] = mnist_parse('train-images-idx3-ubyte', 'train-labels-idx1-ubyte');
images = double(reshape(images, size(images,1)*size(images,2), []));
images = double(images);
% load test data
[images_t, labels_t] = mnist_parse('t10k-images-idx3-ubyte', 't10k-labels-idx1-ubyte');
images_t = double(reshape(images_t, size(images_t,1)*size(images_t,2), []));
images_t = double(images_t);
% svd

[U,S,V] = svd([images],'econ');
[Ut, St, Vt] = svd([images_t], 'econ');
% singular value
figure(1)
subplot(2,1,1)
plot(diag(S),'ko','Linewidth',2)
title('Singular Value Spectrum')
set(gca,'Fontsize',16,'Xlim',[0 800])
xlabel('modes')
ylabel('singluar values')
subplot(2,1,2)
semilogy(diag(S),'ko','Linewidth',2)
set(gca,'Fontsize',16,'Xlim',[0 800])
xlabel('modes')
ylabel('singluar values in log')
%% Projection onto 3 V-modes

for label=0:9
    label_indices = find(labels == label);
    plot3(V(label_indices, 2), V(label_indices, 3), V(label_indices, 5),...
        '.', 'DisplayName', sprintf('%i',label), 'Linewidth', 2)
    hold on
end
figure(2)
xlabel('2nd V-Mode'), ylabel('3rd V-Mode'), zlabel('5th V-Mode')
title('Projection onto V-modes 2, 3, 5')
legend('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')
set(gca,'Fontsize', 14)
%% pick feature
feature = 70;

projection = S*V'; % projection onto principal components: X = USV' --> U'X = SV'
TestMat = U(:, 1:feature)'* images_t;
%% LDA
label_0 =find(labels == 0);
proj_0 = projection(1:feature,label_0);
label_1 = find(labels == 1);
proj_1 = projection(1:feature, label_1);

% Calculate scatter matrices

m0 = mean(proj_0,2);
m1 = mean(proj_1,2);

Sw = 0; % within class variances
for k = 1:size(label_0)
    Sw = Sw + (proj_0(:,k) - m0)*(proj_0(:,k) - m0)';
end
for k = 1:size(label_1)
   Sw =  Sw + (proj_1(:,k) - m1)*(proj_1(:,k) - m1)';
end

Sb = (m0-m1)*(m0-m1)'; % between class
% Find the best projection line

[V2, D] = eig(Sb,Sw); % linear disciminant analysis
[lambda, ind] = max(abs(diag(D)));
w = V2(:,ind);
w = w/norm(w,2);

% Project onto w

v0 = w'*proj_0;
v1 = w'*proj_1;

% Make one digit below the threshold

if mean(v0) > mean(v1)
    w = -w;
    v0 = -v0;
    v1 = -v1;
end

% Find the threshold value

sort0 = sort(v0);
sort1 = sort(v1);

t1 = length(sort0);
t2 = 1;
while sort0(t1) > sort1(t2)
    t1 = t1 - 1;
    t2 = t2 + 1;
end
threshold = (sort0(t1) + sort1(t2))/2;

% Plot histogram of results

figure(5)
subplot(1,2,1)
histogram(sort0,30); hold on, plot([threshold threshold], [0 10],'r')
%set(gca,'Xlim',[-3 4],'Ylim',[0 10],'Fontsize',14)
title('8')
subplot(1,2,2)
histogram(sort1,30); hold on, plot([threshold threshold], [0 10],'r')
%set(gca,'Xlim',[-3 4],'Ylim',[0 10],'Fontsize',14)
title('9')
accuracy_LDA = accuracy_rate_LDA(w, TestMat, labels_t, 0, 1, threshold );


%% a linear classifier that identify 3 arbitrarily picked digits
projection_3 = projection(1:feature, :);
projection_3= projection(find(labels == 0| labels == 1| labels == 2));
labels_3 = labels(find(labels == 0| labels == 1| labels == 2));
test_3 = TestMat(find(labels_t == 0| labels_t == 1| labels_t == 2));
labels_3t = labels_t(find(labels_t == 0| labels_t == 1| labels_t == 2));
class = classify(test_3, projection_3, labels_3, 'linear');
count = 0;
for j = 1:length(test_3)
    if class(j) == labels_3t(j)
        count = count+1;
    end
end
accuracy_3 = count/length(test_3)

%% SVM classifier with training data, labels and test set

Mdl = fitcecoc(projection(1:50, :)', labels);
predict_labels_svm = predict(Mdl, TestMat');
%% prediction with svm classifier
pairs01_svm = accuracy_rate(predict_labels_svm, labels_t, 0, 1);
pairs35_svm = accuracy_rate(predict_labels_svm, labels_t, 3, 5);
%% decision tree classifier
tree = fitctree(projection(1:feature, :)', labels);
predict_labels_tree = predict(tree, TestMat');

%% prediction with decision tree classifier
pairs01_tree = accuracy_rate(predict_labels_tree, labels_t, 0, 1);
pairs35_tree = accuracy_rate(predict_labels_tree, labels_t, 3, 5);
%% Accuracy rate for LDA
function accuracy_LDA = accuracy_rate_LDA(w, TestMat, labels_t, digit1, digit2, threshold)
    pval = w'*TestMat;
    digit1_t = find(labels_t ==digit1);
    digit2_t = find(labels_t == digit2);
    count = 0;
    index = 0;
    for j = 1:length(digit1_t)
        index = index+1;
        if pval(digit1_t(index)< threshold)
            count = count +1;
        end
    end
    index1 = 0;
    for j = 1:length(digit2_t)
        index1 = index1 + 1;
        if pval(digit2_t(index) > threshold)
            count = count +1;
        end
    end
    accuracy_LDA = count/(length(digit1_t)+length(digit2_t));
end
%% function for calculating accuracy rate for desicion tree and SVM classifer
function accuracy = accuracy_rate(predict_labels, labels_t, digit1, digit2)
    predict_digit = predict_labels(find(labels_t== digit1 | labels_t == digit2));
    test_digit = labels_t(find(labels_t== digit1 | labels_t == digit2));
    count = 0;
    for i = 1: length(predict_digit)
        if predict_digit(i) == test_digit(i)
            count = count+1;
        end
    end
    accuracy = count/length(test_digit);
end