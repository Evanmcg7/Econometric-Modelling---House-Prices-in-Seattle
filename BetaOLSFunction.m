function [Beta_hat,CI,t_stat,p_value_t,R2,AdjR2,F_statistic,Fp_value,Skewness,Kurtosis,JB,JBcrit,H,BP_stat,bpcrit,pvalue_BP,DW,VIF] = BetaOLSFunction(X,Y)

% Data - Provided in Main Script
[data,text]=xlsread('houseprice_data_2019.xlsx');
data1 = data(:, :);
names = text;
Y = data1(:, 1);
X = data1(:, 2:end);
[T,N] = size(X);
X = [ones(T,1) X];
K=size(X,2);
df = T-K;

% (a) OLS Estimators
[Beta_hat] = (X'*X)\X'*Y

% (b) Confidence Intervals
Yhat = X*Beta_hat;
residuals = Y - Yhat;
sigma_r = (residuals.'*residuals)/(T-K);
varbeta = sigma_r.*inv(X'*X);
stderr_beta = sqrt(diag(varbeta));
% Calculate Confidence Interval (CI)
CI = [Beta_hat - stderr_beta.*1.96, Beta_hat + stderr_beta.*1.96]

% (c) Statistical Significance
t_stat = abs(Beta_hat./stderr_beta)
p_value_t = tcdf(t_stat, T-N,"upper")*2

% (d) R^2 and Adjusted R^2
Ybar = mean(Y);
R2 = sum((Yhat-Ybar).^2)/sum((Y-Ybar).^2)
AdjR2 = 1-((T-1)/(T-K)*(1-R2))

% (e) F-statistic
regressors_remained = 1;
m = K-regressors_remained;
F_statistic = (R2/(m))/((1-R2)/(T-K))
Fp_value = 1 - fcdf(F_statistic,m,T-K)

% (f) Plot of the Fitted Model
% Plot of Fitted Model
figure(1)
coefficients = polyfit(Yhat,Y,1);
xFit = linspace(min(Yhat),max(Yhat),2000);
yFit = polyval(coefficients,xFit);
subplot(1,2,1)
plot(Yhat,Y,'b*')
hold on;
plot(xFit,yFit,'r-','LineWidth',2);
grid on;
xlabel("Estimated House Prices (giving Line of Best Fit)")
ylabel("Observed House Prices")
title("Plot of the Fitted Model")
hold off

subplot(1,2,2)
scatter(Yhat,residuals)
xlabel("Estimated House Prices")
ylabel("Residuals")
title("Residuals Scatter Plot")
hold off

% (g) Diagnostic tests for residuals
% Normal Distribution
figure(2)
subplot(1,2,1)
qqplot(residuals)
title('Normal Q-Q');
ylabel('Residuals');
subplot(1,2,2)
histogram(residuals);
title('Histogram of Residuals');
xlabel('Residuals'); ylabel('Frequency');

hold off

Skewness = skewness(residuals)
Kurtosis = kurtosis(residuals)

JB = T*((skewness(residuals)^2)/6 + ((kurtosis(residuals)-3)^2)/24)
JBcrit=chi2inv(0.95,2)
if JB > JBcrit      % H = 1; reject H0
    H = 1
elseif JB < JBcrit  % H = 0 means Normal Distribution present
    H = 0
end

% Heteroskedasticity
RSS = (residuals.'*residuals);
sigmaeps = (1./T).*RSS;
epsnew = (residuals.^2)./sigmaeps - 1;
Beta_BP = inv(X.'*X)*X.'*epsnew;
epsfitted = X*Beta_BP;
BP_stat = sum(epsfitted.^2)./2
bpcrit = chi2inv(0.95,m)
pvalue_BP = 1-chis_prb(BP_stat,m)

% Serial Correlation
DW=sum(diff(residuals,1).^2)./sum(residuals.^2)

% (h) Multicollinearity
R0 = corrcoef(X(:,2:end));
VIF_X = diag(inv(R0))';
VIF = array2table(VIF_X)

end