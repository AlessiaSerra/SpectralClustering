clear;
close all; 
clc;

n_inner = 100;
n_mid = 200; 
n_outer = 300; 
theta_in = 2*pi*rand(n_inner,1); 
r1 = 1 + 0.05*randn(n_inner,1);
theta_mid = 2*pi*rand(n_mid,1);   
r2 = 3 + 0.05*randn(n_mid,1);
theta_out = 2*pi*rand(n_outer,1);
r3 = 5 + 0.05*randn(n_outer,1);
X = [ [r1.*cos(theta_in), r1.*sin(theta_in)]; 
      [r2.*cos(theta_mid), r2.*sin(theta_mid)];
      [r3.*cos(theta_out), r3.*sin(theta_out)] ];
k = 3; 
 
sigma = 0.3;
W = exp(-pdist2(X, X).^2 / (2*sigma^2));
W = W - diag(diag(W));
D = diag(sum(W, 2));
% Unnormalized case
L_unnorm = D - W;
[V_unnormalized, lambda] = eig(L_unnorm); 
[~, idx_sort] = sort(diag(lambda), 'ascend'); 
V_un = V_unnormalized(:, idx_sort(1:k));
idx_un = kmeans(V_un, k);

%  Normalized case
D_inv_sqrt = diag(1 ./ sqrt(diag(D)));
L_sym = eye(size(W)) - D_inv_sqrt * W * D_inv_sqrt;
[V_normalized, lambda_sym] = eig(L_sym); 
[~, idx_sort_sym] = sort(diag(lambda_sym), 'ascend');
V_sym = V_normalized(:, idx_sort_sym(1:k));
Phi = V_sym ./ sqrt(sum(V_sym.^2, 2)); 
idx_norm = kmeans(Phi, k);

%  EMBEDDING
figure('Name', 'Spectral Embedding', 'Position', [100 100 1200 450]);

%  Unnormalized Embedding
subplot(1,2,1);
if k == 2
    gscatter(V_un(:,1), V_un(:,2), idx_un);
    xlabel('v1'); ylabel('v2');
else
    scatter3(V_un(:,1), V_un(:,2), V_un(:,3), 30, idx_un, 'filled');
    xlabel('v1'); ylabel('v2'); zlabel('v3');
    view(3);
end
title('Embedding: Unnormalized');
grid on;

%  Normalized Embedding
subplot(1,2,2);
if k == 2
    gscatter(Phi(:,1), Phi(:,2), idx_norm);
    xlabel('Normalized Eigenvector 1'); ylabel('Normalized Eigenvector 2');
else
    scatter3(Phi(:,1), Phi(:,2), Phi(:,3), 30, idx_norm, 'filled');
    xlabel('\phi_1'); ylabel('\phi_2'); zlabel('\phi_3');
    view(3);
end
title('Embedding: Normalized');
grid on;

figure('Position', [100 100 1200 450]);
subplot(1,2,1); gscatter(X(:,1), X(:,2), idx_un); axis equal; grid on;
title('Unnormalized '); 
subplot(1,2,2); gscatter(X(:,1), X(:,2), idx_norm); axis equal; grid on;
title('Normalized');

