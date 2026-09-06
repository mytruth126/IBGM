import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
import matplotlib.pyplot as plt

def bayesian_sindy_fit_refit(x, t, dt=None, noise_std=0.0, max_power=3,
                             max_terms = 5, num_warmup=500, num_samples=2000, seed=0,
                             coef_threshold=0.1, h=0):
    """
    两步贝叶斯SINDy + 趋势外推
    h: 外推步数
    """
    key = jax.random.PRNGKey(seed)
    x = jnp.array(x)
    t = jnp.array(t)
    N = len(t)
    if dt is None:
        dt = t[1] - t[0]

    # -----------------------------
    # 数值微分
    # -----------------------------
    dxdt = (x[1:] - x[:-1]) / dt
    if noise_std > 0.0:
        dxdt = dxdt + noise_std * jax.random.normal(key, dxdt.shape)

    # -----------------------------
    # 构建特征矩阵 X
    # -----------------------------
    X_list = [jnp.ones_like(x[:-1])]  # 常数项
    terms = ["1"]

    # x^p
    for p in range(1, max_power + 1):
        X_list.append(x[:-1] ** p)
        terms.append(f"x^{p}")
    
    # t^p
    for p in range(1, max_power + 1):
        X_list.append(t[:-1] ** p)
        terms.append(f"t^{p}")

    # (x/t)^p
    for p in range(1, max_power + 1):
        xt_ratio = jnp.where(t[:-1]==0, 0.0, x[:-1]/t[:-1])
        X_list.append(xt_ratio ** p)
        terms.append(f"(x/t)^{p}")

    X = jnp.column_stack(X_list)
    y = dxdt

    # -----------------------------
    # 第一步：稀疏贝叶斯筛选结构
    # -----------------------------
    def sparse_model(X, y=None):
        n_features = X.shape[1]
        beta = numpyro.sample("beta", dist.Laplace(0, 0.5).expand([n_features]))
        sigma = numpyro.sample("sigma", dist.HalfCauchy(0.1))
        mu = X @ beta
        numpyro.sample("obs", dist.Normal(mu, sigma), obs=y)

    nuts = NUTS(sparse_model)
    mcmc = MCMC(nuts, num_warmup=num_warmup, num_samples=num_samples)
    mcmc.run(key, X, y)
    samples = mcmc.get_samples()

    beta_mean = np.array(samples["beta"].mean(axis=0))
    beta_low = np.array(np.percentile(samples["beta"], 0.5, axis=0))
    beta_high = np.array(np.percentile(samples["beta"], 99.5, axis=0))
    significant_mask = (beta_low * beta_high > 0) & (np.abs(beta_mean) > coef_threshold)

    # -----------------------------
    # 结构项数量限制 - 新增部分
    # -----------------------------
    # 获取显著项的系数绝对值
    significant_indices = np.where(significant_mask)[0]
    if len(significant_indices) > 0:
        significant_coefs_abs = np.abs(beta_mean[significant_mask])
        
        # 按系数绝对值排序
        sorted_indices = significant_indices[np.argsort(significant_coefs_abs)[::-1]]
        
        # 最多保留5个项（包括常数项）

        if len(sorted_indices) > max_terms:
            print(f"结构项过多 ({len(sorted_indices)}个)，只保留前{max_terms}个最重要项")
            sorted_indices = sorted_indices[:max_terms]
        
        # 确保至少保留1个非常数项
        non_constant_terms = [idx for idx in sorted_indices if terms[idx] != "1"]
        if len(non_constant_terms) == 0 and len(sorted_indices) > 1:
            # 如果没有非常数项，从剩余项中选择一个最重要的非常数项
            remaining_non_constant = [idx for idx in significant_indices if terms[idx] != "1"]
            if len(remaining_non_constant) > 0:
                best_non_constant = remaining_non_constant[np.argmax(np.abs(beta_mean[remaining_non_constant]))]
                sorted_indices = sorted_indices[:max_terms-1] + [best_non_constant]
                print(f"添加重要非常数项: {terms[best_non_constant]}")
        
        # 更新显著掩码
        significant_mask = np.zeros_like(significant_mask, dtype=bool)
        significant_mask[sorted_indices] = True
    else:
        # 如果没有显著项，选择绝对值最大的2个项（至少包含1个非常数项）
        print("没有显著项，选择最重要的2个项")
        top_k = 2
        all_indices = np.argsort(np.abs(beta_mean))[-top_k:]
        
        # 确保至少有一个非常数项
        non_constant_indices = [idx for idx in all_indices if terms[idx] != "1"]
        if len(non_constant_indices) == 0:
            # 如果选中的都是常数项，替换其中一个为最重要的非常数项
            non_constant_all = [idx for idx in range(len(terms)) if terms[idx] != "1"]
            if len(non_constant_all) > 0:
                best_non_constant = non_constant_all[np.argmax(np.abs(beta_mean[non_constant_all]))]
                all_indices = [all_indices[0], best_non_constant]
        
        significant_mask = np.zeros_like(significant_mask, dtype=bool)
        significant_mask[all_indices] = True

    # -----------------------------
    # 第二步：只保留显著项重新拟合参数
    # -----------------------------
    X_significant = X[:, significant_mask]
    terms_significant = [term for term, sig in zip(terms, significant_mask) if sig]

    print(f"\n最终选择的项 ({len(terms_significant)}个): {terms_significant}")

    def refit_model(X, y=None):
        n_features = X.shape[1]
        beta = numpyro.sample("beta", dist.Normal(0, 1).expand([n_features]))
        sigma = numpyro.sample("sigma", dist.HalfCauchy(0.1))
        mu = X @ beta
        numpyro.sample("obs", dist.Normal(mu, sigma), obs=y)

    nuts_refit = NUTS(refit_model)
    mcmc_refit = MCMC(nuts_refit, num_warmup=num_warmup, num_samples=num_samples)
    mcmc_refit.run(key, X_significant, y)
    samples_refit = mcmc_refit.get_samples()
    
    beta_samples_refit = samples_refit["beta"]
    beta_mean_refit = np.array(beta_samples_refit.mean(axis=0))
    beta_low_refit = np.array(np.percentile(beta_samples_refit, 0.5, axis=0))
    beta_high_refit = np.array(np.percentile(beta_samples_refit, 99.5, axis=0))

    # -----------------------------
    # 构建辨识公式
    # -----------------------------
    identified_terms = []
    for coef, name in zip(beta_mean_refit, terms_significant):
        identified_terms.append(f"{coef:.4f}·{name}")
    identified_formula = " + ".join(identified_terms) if identified_terms else "0"

    # -----------------------------
    # 用辨识公式求解 x(t) - 数值稳定性版本
    # -----------------------------
    def identified_dxdt(x_val, t_val):
        val = 0.0
        for coef, name in zip(beta_mean_refit, terms_significant):
            try:
                if name == "1":
                    val += coef
                elif name.startswith("x^"):
                    p = float(name.split("^")[1])
                    # 限制x的幂次计算范围
                    x_power = np.clip(x_val**p, -1e10, 1e10)
                    val += coef * x_power
                elif name.startswith("t^"):
                    p = float(name.split("^")[1])
                    # 限制t的幂次计算范围
                    t_power = np.clip(t_val**p, -1e10, 1e10)
                    val += coef * t_power
                elif name.startswith("(x/t)^"):
                    p = float(name.split("^")[1])
                    # 避免除零和数值溢出
                    safe_t = t_val if t_val != 0 else 1e-10
                    ratio = x_val / safe_t
                    ratio = np.clip(ratio, -1e6, 1e6)  # 限制比值范围
                    ratio_power = np.clip(ratio**p, -1e10, 1e10)
                    val += coef * ratio_power
            except (ValueError, ZeroDivisionError, OverflowError):
                # 如果计算出现错误，跳过该项
                continue
                
        # 限制导数值的范围
        return np.clip(val, -1e6, 1e6)

    # -----------------------------
    # RK4积分（含外推）- 添加数值稳定性
    # -----------------------------
    x_identified = [float(x[0])]  # 确保初始值为浮点数
    t_extended = np.concatenate([t, t[-1] + dt * np.arange(1, h+1)])
    
    for i in range(len(t_extended)-1):
        current_x = x_identified[i]
        current_t = t_extended[i]
        
        # 检查当前值是否有效
        if not np.isfinite(current_x):
            # 如果出现无效值，使用线性外推
            if i > 0:
                current_x = 2 * x_identified[i-1] - x_identified[i-2] if i > 1 else x_identified[i-1]
            else:
                current_x = float(x[0])
            x_identified[i] = current_x
        
        try:
            # 尝试RK4积分
            k1 = identified_dxdt(current_x, current_t)
            k2 = identified_dxdt(current_x + 0.5*dt*k1, current_t + 0.5*dt)
            k3 = identified_dxdt(current_x + 0.5*dt*k2, current_t + 0.5*dt)
            k4 = identified_dxdt(current_x + dt*k3, current_t + dt)
            
            x_next = current_x + dt*(k1 + 2*k2 + 2*k3 + k4)/6
            
            # 检查结果是否有效
            if not np.isfinite(x_next):
                # 如果RK4失败，使用欧拉方法
                dxdt_val = identified_dxdt(current_x, current_t)
                x_next = current_x + dt * dxdt_val
                
            # 最终有效性检查
            if not np.isfinite(x_next):
                # 如果仍然无效，使用线性外推
                if i > 0:
                    x_next = 2 * x_identified[i-1] - x_identified[i-2] if i > 1 else x_identified[i-1]
                else:
                    x_next = float(x[0])
                    
        except (ValueError, ZeroDivisionError, OverflowError):
            # 如果出现异常，使用线性外推
            if i > 0:
                x_next = 2 * x_identified[i-1] - x_identified[i-2] if i > 1 else x_identified[i-1]
            else:
                x_next = float(x[0])
        
        # 限制x的范围避免过大或过小
        x_next = np.clip(x_next, -1e10, 1e10)
        x_identified.append(float(x_next))
    
    x_identified = np.array(x_identified)

    return {
        "t": t_extended,
        "x_true": np.concatenate([x, [np.nan]*h]) if h>0 else x,
        "x_identified": x_identified,
        "beta_mean": beta_mean_refit,
        "beta_samples": beta_samples_refit,
        "beta_low": beta_low_refit,
        "beta_high": beta_high_refit,
        "identified_formula": identified_formula,
        "terms_significant": terms_significant
    }