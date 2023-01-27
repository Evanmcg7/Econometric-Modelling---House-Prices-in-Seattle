%% 1. Importing Data
% House-Price in the USA at 1/1/2019 (pre-cleaned) - using 8 Regressors
% Evaluate this before running the 'OLS Regression Function'
[data,text]=xlsread('houseprice_data_2019.xlsx');
data1 = data(:, :);
names = text;
Y = data1(:, 1);
X = data1(:, 2:end);
[T,N] = size(X);
X = [ones(T,1) X];
K=size(X,2);
df = T-K;

%% 2. OLS Regression Function [Code also provided at end of this script]
BetaOLSFunction(X,Y)

%% 3. ARIMA Forecasting Model
%% 1. Import Data
[data,text] = xlsread('FREDQ_DATA.xlsx');

data1 = data(3:end,:);
names = text(1,2:end);
date = text(4:end,1);

% Data of Interest
endsample = find(contains(date,'01/09/2022'));
DATAFINAL = data1(1:endsample,:);

idx = [1 12];
DATASELECT = DATAFINAL(:,idx);
LABEL = names(idx);

time = 1959+(3/12):(3/12):2022+(09/12);
ylabs = ["Index" "Index"];

% Plot Raw Time Series
P1=figure(1);
for i=1:size(DATASELECT,2)  % alternatively cols(DATASELECT)
    subplot(size(DATASELECT,2),1,i)
    plot(time,DATASELECT(:,i),'LineWidth',1.3,'Color','k')
    axis tight
    ylabel(ylabs(i)); xlabel("Years")
    title(LABEL(i))
end
hold off

% Data Selection for Correlograms
GDP = DATASELECT(:,1);
GovExp = DATASELECT(:,2);
ts1 = [GDP,GovExp];
figure()
plot(time,ts1)
legend('GDP','GovExp')
hold off

% ACF & PACF
figure
subplot(2,1,1)
autocorr(GDP)
subplot(2,1,2)
parcorr(GDP)

% Augmented Dickey-Fuller Test
disp('    T-stat    Pval (Lag = Auto (BIC), No Deterministic)')
[ts, pv] = augdfautolag(GDP,0,20,'BIC');
disp([ts,pv]);

disp('    T-stat    Pval (Lag = Auto (BIC), Intercept)')
[ts, pv] = augdfautolag(GDP,1,20,'BIC');
disp([ts,pv]);

disp('    T-stat    Pval (Lag = Auto (BIC), Intercept & Trend)')
[ts, pv] = augdfautolag(GDP,2,20,'BIC');
disp([ts,pv]);



%% 2. Data Selection + Transformation for ARIMA Forecasting
% Transformation for Stationarity [FRED-MD Suggested]
tcode = [5 5];
DATATRANSFORM=getdatatransform(DATASELECT,tcode);

plotx1(DATATRANSFORM,tcode,LABEL,time) % Trend Removed
hold off

GDP_R = DATATRANSFORM(:,1);
ts1 = [GDP_R];
figure()
plot(time,ts1)
legend('GDP')
hold off

Y = GDP_R(1:150,:);

T = 120;

%% 3. ACF & PACF
figure
subplot(2,1,1)
autocorr(Y)
subplot(2,1,2)
parcorr(Y)

%% 4. ARIMA Model Estimation - Different Lag Combinations
% Estimate 4 ARIMA Models with different lags combinations
LOGL = zeros(4,4); %Initialize
PQ = zeros(4,4);
for p = 1:4
    for q = 1:4
        mod = arima(p,0,q);
        [fit,~,logL] = estimate(mod,Y);
        LOGL(p,q) = logL;
        PQ(p,q) = p+q;
     end
end

% Calculate BIC - Information Criterion
LOGL = reshape(LOGL,16,1);
PQ = reshape(PQ,16,1);
[~,bic] = aicbic(LOGL,PQ+1,100);
reshape(bic,4,4)
% Choose ARMA(2,4)

%% 5. ARMA Residual Diagnostics
% ARMA(2,4) as indicated by BIC
%Remove NaN
Y_T = GDP_R(2:end,:);

[parameters, ll, errors] = armaxfilter(Y_T,0,2,4);
disp('Parameters')
disp(parameters)
% Standardized errors are simpler to use since they are scale free
stderrors = errors / std(errors);

% ACF & PACF Plots of Residuals
figure()
plot(stderrors)
sacf(stderrors,24)
spacf(stderrors,24)
hold off

% Autocorrelation at Lags 1 & 24
% Implement Ljung-Box Test
disp('Ljung-Box Test (Test stat, p-value)' )
[ts,pv]=ljungbox(stderrors,12);
disp([ts,pv])

% P-value is High [No Serial Correlation]

%% 6. Forecast pseudo out-of-sample with ARIMA Model
% According to BIC, we select an ARMA(2,4) Model
ToEstMdlm1 = arima('ARLags',2,'MALags',4);
[EstMdlm1, logL1] = estimate(ToEstMdlm1,Y(1:T));

h = 30; % Horizons
[YFm1,YMSEm1] = forecast(EstMdlm1,h,'Y0',Y(1:T));

% Forecast Error
EFm1 = Y(T+1:T+h)-YFm1;

% Root Mean Square Forecast Error (RMSE)
EF2m1=EFm1.^2;
RMSFEm1 = [sum(EF2m1)]/h;

figure
h1 = plot(Y,'Color',[.7,.7,.7]);
hold on
h2 = plot(121:121+h-1,YFm1,'b','LineWidth',2);
h3 = plot(121:121+h-1,YFm1 + 1.96*sqrt(YMSEm1),'r:',...
		'LineWidth',2);
plot(121:121+h-1,YFm1 - 1.96*sqrt(YMSEm1),'r:','LineWidth',2);
legend([h1 h2 h3],'Observed','Forecast',...
		'95% Confidence Interval','Location','NorthWest');
title(['30-Period Forecasts and Approximate 95% '...
			'Confidence Intervals'])
hold off

%% 7. Choose ARMA(3,3) and use to forecast GDP
ToEstMdlm2 = arima('ARLags',3,'MALags',3);
EstMdlm2 = estimate(ToEstMdlm2,Y(1:T));

h = 30; %%%horizons
[YFm2,YMSEm2] = forecast(EstMdlm2,h,'Y0',Y(1:T));

%%%% Forecast error
EFm2 = Y(T+1:T+h)-YFm2;

%%%% Root Mean Square Forecast Error (RMSE)

EF2m2=EFm2.^2;

RMSFEm2 = [sum(EF2m2)]/h;

figure
h1 = plot(Y,'Color',[.7,.7,.7]);
hold on
h2 = plot(121:121+h-1,YFm2,'b','LineWidth',2);
h3 = plot(121:121+h-1,YFm2 + 1.96*sqrt(YMSEm2),'r:',...
		'LineWidth',2);
plot(121:121+h-1,YFm2 - 1.96*sqrt(YMSEm2),'r:','LineWidth',2);
legend([h1 h2 h3],'Observed','Forecast',...
		'95% Confidence Interval','Location','NorthWest');
title(['30-Period Forecasts and Approximate 95% '...
			'Confidence Intervals'])
hold off

%% 8. Compare ARIMA Models
RMSE = [RMSFEm1, RMSFEm2]

% If we compare RMSFEm1 (0.3353) and RMSFEm2 (0.3282) we select ARMA(3,3)

%% 9. Statistically Different
% We can use the Diebold-Mariano test to check if the two forecasts are
% statistically different or not
% Under H0: EFm1 = EFm2 (where EFm1 and EFm2 are the Mean Square Forecast 
% Error for ARMA(2,4) and ARMA(3,3)
 [DMstat, pvalue] = dmtest1(EFm1, EFm2, h)

 % according to the pvalue, we fail to reject H0 hence (even if RMSFE is
 % lower for ARMA(2,4) than for ARMA(3,3)) both models have the same
 % forecasting performance. We don't have a superior model in prediction
 % terms.

%% 10. Forecast Out-of-Sample [Random Walk]
ToEstMdlm3 = arima('ARLags',1,'D',1,'MALags',1);
EstMdlm3 = estimate(ToEstMdlm3,GDP);

h = 30; % Horizons
[YFm3,YMSEm3] = forecast(EstMdlm3,h,'Y0',(GDP));

YFm3(2)

figure
h1 = plot(GDP,'Color',[.7,.7,.7]);
hold on
h2 = plot(256:256+h-1,YFm3,'b','LineWidth',2);
h3 = plot(256:256+h-1,YFm3 + 1.96*sqrt(YMSEm3),'r:',...
		'LineWidth',2);
plot(256:256+h-1,YFm3 - 1.96*sqrt(YMSEm3),'r:','LineWidth',2);
legend([h1 h2 h3],'Observed','Forecast',...
		'95% Confidence Interval','Location','NorthWest');
title(['30-Period Forecasts and Approximate 95% '...
			'Confidence Intervals'])
hold off




%% Assignment OLS Script [Code used in Function with explanations]
%% 1. OLS Estimation [Part 1 (a)]
% The model can be written as follows
%   Y   =   X    *   beta    +  EPSILON
% (T*1)   T*(N+1)  (N+1)*1       (T*1)

Beta_hat = (X'*X)\X'*Y
% Estimated 'Y'
Yhat = X*Beta_hat;

% Residuals 
residuals = Y - Yhat; 

% Variance & Standard Error of Residuals
sigma_r = (residuals.'*residuals)/(T-K); % Residual Variance Estimator
r_stderr = sqrt(diag(sigma_r)); % Residual Variance Std. Error

% Variance & Standard Error of OLS Estimators
varbeta = sigma_r.*inv(X'*X); % Var/Covar Matrix
stderr_beta = sqrt(diag(varbeta)); % Standard Errors

%% 2. Confidence Intervals [Part 1(b)]
alpha = 0.05 % Level of Significance
t_critical = tinv(1-alpha/2,T-K-1); % Critical 't' = 1.96
CI = [Beta_hat - stderr_beta.*1.96, Beta_hat + stderr_beta.*1.96]

%% 3. Statistical Significance of Beta Coefficients [Part 1 (c)]
% H0 (null hypothesis): beta_i = 0 (for i=0,...,8)
% H1 (alternative hypothesis): beta_i ~= 0 (for i=0,...,8)

t_stat = abs(Beta_hat./stderr_beta) % empirical t-statistics
t_critical = tinv(1-alpha/2,T-N)
p_value_t = tcdf(t_stat, T-N,"upper")*2
%% 4. Goodness of Fit [Part 1 (d)]
% R-squared
Ybar = mean(Y);
R2 = sum((Yhat-Ybar).^2)/sum((Y-Ybar).^2)
% Adjusted R-squared
AdjR2 = 1-((T-1)/(T-K)*(1-R2))
%% 5. F-statistic
% H0; Beta_2 = Beta_3 = Beta_4 = Beta_5 ... Beta_8 = 0]
% H1; Beta_i ~= 0 (for i=2,...,8]

% Restricted Regression
regressors_remained = 1;
m = K-regressors_remained; % the number of restrictions

% Empirical 'F-Statistic'
F_statistic = (R2/(m))/((1-R2)/(T-K))

% P-Value
Fp_value = 1 - fcdf(F_statistic,m,T-K)


%% 6. Plot of the Fitted Model [Part 1 (f)]
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

%% 7. Diagnostic Test for Normality [Part 1 (g)]
% Plotting Residuals - visual check of normality assumption
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

% Jarque-Bera Test
% H0; We assume that residuals are Normally Distributed
% H1; We assume that residuals are not Normally Distributed
Skewness = skewness(residuals)
Kurtosis = kurtosis(residuals)

JB = T*((skewness(residuals)^2)/6 + ((kurtosis(residuals)-3)^2)/24)
JBcrit=chi2inv(0.95,2)

if JB > JBcrit      % H = 1; reject H0
    H = 1
elseif JB < JBcrit  % H = 0 means Normal Distribution present
    H = 0
end

%% 8. Residuals Diagnostic Test for Heteroskedasticity [Part 1 (g)]
% Breusch-Pagan Test
% H0; Delta2 = Delta3 ... Delta 9 = 0
% H1; there is heteroskedasticity present
RSS = (residuals.'*residuals);
sigmaeps = (1./T).*RSS;
epsnew = (residuals.^2)./sigmaeps - 1;
Beta_BP = inv(X.'*X)*X.'*epsnew;
epsfitted = X*Beta_BP;

BP_stat = sum(epsfitted.^2)./2
bpcrit = chi2inv(0.95,m)
pvalue_BP = 1-chis_prb(BP_stat,m)

% There are no square terms or cross-terms in regression
% Breusch-Pagan is satisfactory to show Heteroskedasticity
% Heteroskedasticity present

%% 9. Residuals Diagnostic Test for Serial Correlation [Part 1 (g)]
% Durbin-Watson test
% H0: rho = 0
% H1: eps_{t} = rho*eps_{t-1} + u_{t}, with rho different from 0.
DW=sum(diff(residuals,1).^2)./sum(residuals.^2)

% DW is approximately equal to 2. Implies NO SERIAL CORRELATION

% Checking with Critical Values for Alpha = 0.05 & k=9 [inc. intercept]
dL = 1.917
dU = 1.936
Four_dU = 4 - dU
Four_dL = 4 - dL

% dU < DW < 2
% There is NO Serial Correlation0

%% 10. Testing for Multicollinearity [Part 1 (h)]
R0 = corrcoef(X(:,2:end));
VIF = diag(inv(R0))';
VIF = array2table(VIF)
