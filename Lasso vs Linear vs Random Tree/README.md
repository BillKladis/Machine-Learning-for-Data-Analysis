# Regression Comparison — Software Engineer Salary

Dataset: Software engineer salaries with experience, country, education, languages and frameworks  
Algorithms: Linear Regression, Lasso, Random Forest Regressor  
Task: Predict salary in USD

## Results
| Model | R2 | RMSE |
|-------|----|------|
| Linear Regression | - | - |
| Lasso | - | - |
| Random Forest | - | - |

## Plots
![R2 comparison](plot_r2_comparison.png)
![RMSE comparison](plot_rmse_comparison.png)
![Predicted vs actual](plot_predicted_vs_actual.png)
![Residuals](plot_residuals.png)
![Lasso coefficients](plot_lasso_coefficients.png)
![Random Forest importance](plot_rf_importance.png)

## Key concepts covered
- Feature engineering on comma-separated multi-label columns
- Lasso automatic feature selection via lambda penalization
- Why Random Forest does not need scaling
- R2 and RMSE as regression metrics
- Residual analysis to diagnose model behavior