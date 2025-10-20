# Linear Regression from Scratch: Gradient Descent Implementation

This project implements a **Linear Regression** model entirely from scratch using Python and the NumPy library. It utilizes the **Gradient Descent (GD)** algorithm to find the optimal coefficients (weights $w$ and bias $b$) that minimize the Mean Squared Error (MSE) cost function.

The model was trained on the classic **Boston House Price Dataset** (predicting the median value of owner-occupied homes - `MEDV`).

## Key Results

After tuning the hyperparameters, the model demonstrated strong performance on the test set:

| Metric | Value |
| :--- | :--- |
| **R² Score** | **0.743** |
| Mean Absolute Error (MAE) | 3.299 |
| Mean Squared Error (MSE) | 19.959 |
| Root Mean Squared Error (RMSE) | 4.468 |

The **R² score of 0.743** indicates that the model explains approximately 74.3% of the variance in the house prices, which is a significant improvement over the initial low score of $0.226$ (from previous runs) and demonstrates effective convergence of the custom Gradient Descent algorithm.

## Implementation Details

The core of the project is the `Linear_Regression` class, which implements the hypothesis function, the cost function derivative, and the parameter update rule.

### 1. Model Class (`Linear_Regression`)

The model uses the standard linear equation:
$$h(X) = X \cdot w + b$$

The parameters are updated using the Batch Gradient Descent rule derived from the Mean Squared Error (MSE) cost function:
$$J(w, b) = \frac{1}{2m} \sum_{i=1}^{m} (h(x^{(i)}) - y^{(i)})^2$$

The update rules for the weights ($w$) and bias ($b$) are:
$$w \leftarrow w - \alpha \cdot \frac{\partial J}{\partial w}$$
$$b \leftarrow b - \alpha \cdot \frac{\partial J}{\partial b}$$

The partial derivatives (gradients) are calculated as:
$$\frac{\partial J}{\partial w} = \frac{1}{m} X^T \cdot (Xw + b - y)$$
$$\frac{\partial J}{\partial b} = \frac{1}{m} \sum_{i=1}^{m} (Xw + b - y)$$

***Note:** The implementation in the provided code includes a factor of $\mathbf{2}$ in the gradient calculation, which is equivalent to using a slightly adjusted learning rate ($\alpha/2$) for the final step:

```python
# Custom Gradient Calculation (effectively using the standard rule times 2)
y_prediction = self.predict(self.X)
dw = -(2 * (self.X.T).dot(self.y - y_prediction)) / self.m # weights
db = -2 * np.sum(self.y - y_prediction) / self.m  # bias
```

### 2. Hyperparameter Tuning

The successful run was achieved using the following hyperparameters:

- **`learning_rate`**: **$0.01$**
    
- **`no_of_iterations`**: **$1000$**
    

These values were sufficient to ensure the model converged to a low error state quickly on the _standardized_ data.

### 3. Data Preprocessing

Crucially, **Feature Scaling** was performed using `StandardScaler` on both the training and testing data. This is essential for the Gradient Descent algorithm to converge efficiently and prevent features with larger scales (like `TAX` or `B`) from dominating the learning process.

## Visual Evaluation

The **Predicted vs Actual** plot shows a strong linear relationship between the model's predictions and the true target values, with most points tightly clustered around the red line of perfect prediction. This confirms the validity of the $\mathbf{R^2}$ score.

The **Residuals Plot** shows a random scatter of errors around the zero line, with no discernible pattern. This indicates that the linear model assumptions are reasonably met and that the model is **not systematically biased** (no clear under- or over-estimation across the prediction range).

## Conclusion

This project successfully verifies the principles of Linear Regression and Batch Gradient Descent through a clean, custom implementation. The high $\mathbf{R^2}$ score confirms that, with proper feature scaling and hyperparameter selection, the custom algorithm effectively minimized the MSE and found a reliable best-fit line for the data.