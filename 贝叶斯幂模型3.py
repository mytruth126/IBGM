import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
import matplotlib.pyplot as plt

def bayesian_sindy_fit_refit(x, t, dt=None, noise_std=0.0, max_power=3,
                             num_warmup=600, num_samples=1200, seed=0,
                             coef_threshold=0.1, h=0):
    """
    用 ARD 风格贝叶斯稀疏回归做两步贝叶斯SINDy + 趋势外推
    h: 外推步数
    返回字典（含识别公式、参数后验/区间、模拟值等）
    """
    # -----------------------------
    # 准备
    # -----------------------------
    key = jax.random.PRNGKey(seed)
    key_sparse, key_refit = jax.random.split(key, 2)
    x = jnp.array(x)
    t = jnp.array(t)
    N = len(t)
    if dt is None:
        dt = float(t[1] - t[0])

    # -----------------------------
    # 数值微分（以 forward difference）
    # -----------------------------
    dxdt = (x[1:] - x[:-1]) / dt
    if noise_std > 0.0:
        dxdt = dxdt + noise_std * jax.random.normal(key_sparse, dxdt.shape)

    # -----------------------------
    # 构建特征矩阵 X 与列名 terms
    # 使用 x[:-1], t[:-1] 与 dxdt 长度一致
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

    # (x/t)^p （避免 t==0）
    xt_ratio = jnp.where(t[:-1] == 0, 0.0, x[:-1] / t[:-1])
    for p in range(1, max_power + 1):
        X_list.append(xt_ratio ** p)
        terms.append(f"(x/t)^{p}")

    X = jnp.column_stack(X_list)  # shape (N-1, n_features)
    y = dxdt
    n_features = X.shape[1]

    # -----------------------------
    # 第一步：ARD 稀疏贝叶斯筛选结构
    # alpha ~ Gamma, w ~ Normal(0, 1/sqrt(alpha))
    # -----------------------------
    def ard_model(X, y=None):
        # alpha: per-feature precision (Gamma)
        alpha = numpyro.sample("alpha", dist.Gamma(1.0, 1.0).expand([n_features]).to_event(1))
        # w: Normal(0, 1/sqrt(alpha))
        w = numpyro.sample("w", dist.Normal(jnp.zeros(n_features), 1.0 / jnp.sqrt(alpha)).to_event(1))
        sigma = numpyro.sample("sigma", dist.Exponential(1.0))
        mu = X @ w
        numpyro.sample("obs", dist.Normal(mu, sigma), obs=y)

    nuts = NUTS(ard_model)
    mcmc = MCMC(nuts, num_warmup=num_warmup, num_samples=num_samples)
    mcmc.run(key_sparse, X, y)
    samples = mcmc.get_samples()

    # 后验统计（w）
    w_samples = np.array(samples["w"])  # shape (n_samples, n_features)
    w_mean = w_samples.mean(axis=0)
    w_q025 = np.percentile(w_samples, 2.5, axis=0)
    w_q975 = np.percentile(w_samples, 97.5, axis=0)

    # 判定显著：95% CI 不包含 0 且均值绝对值超过阈值
    significant_mask = ((w_q025 > 0) | (w_q975 < 0)) & (np.abs(w_mean) > coef_threshold)

    # -----------------------------
    # 若没有显著项，直接返回零模型
    # -----------------------------
    if not significant_mask.any():
        identified_formula = "0"
        # 构建 t_extended 与 x_identified（仅用初值前向积分0）
        t_extended = np.concatenate([np.array(t), np.array(t[-1] + dt * np.arange(1, h+1))]) if h>0 else np.array(t)
        x_identified = np.empty(len(t_extended))
        x_identified[:] = x[0]  # 常数保持初值（或你可改为 NaN）
        return {
            "t": t_extended,
            "x_true": np.concatenate([np.array(x), np.array([np.nan]*h)]) if h>0 else np.array(x),
            "x_identified": x_identified,
            "beta_mean": np.array([]),
            "beta_samples": np.array([]),
            "beta_low": np.array([]),
            "beta_high": np.array([]),
            "identified_formula": identified_formula,
            "terms_significant": []
        }

    # -----------------------------
    # 第二步：只保留显著项重新拟合参数（使用常规高斯先验更稳定）
    # -----------------------------
    X_significant = X[:, significant_mask]
    terms_significant = [term for term, sig in zip(terms, significant_mask) if sig]
    n_sig = X_significant.shape[1]

    def refit_model(Xs, y=None):
        beta = numpyro.sample("beta", dist.Normal(jnp.zeros(n_sig), 5.0).to_event(1))
        sigma = numpyro.sample("sigma", dist.Exponential(1.0))
        mu = Xs @ beta
        numpyro.sample("obs", dist.Normal(mu, sigma), obs=y)

    nuts_refit = NUTS(refit_model)
    mcmc_refit = MCMC(nuts_refit, num_warmup=num_warmup, num_samples=num_samples)
    mcmc_refit.run(key_refit, X_significant, y)
    samples_refit = mcmc_refit.get_samples()

    beta_samples_refit = np.array(samples_refit["beta"])  # (n_samples, n_sig)
    beta_mean_refit = beta_samples_refit.mean(axis=0)
    beta_q025_refit = np.percentile(beta_samples_refit, 2.5, axis=0)
    beta_q975_refit = np.percentile(beta_samples_refit, 97.5, axis=0)

    # -----------------------------
    # 构建辨识公式字符串
    # -----------------------------
    def format_formula(coefs, names):
        parts = []
        for c, n in zip(coefs, names):
            parts.append(f"{c:.4f}·{n}")
        return " + ".join(parts) if parts else "0"

    identified_formula = format_formula(beta_mean_refit, terms_significant)

    # -----------------------------
    # 用辨识公式求解 dx/dt（用于 RK4）
    # -----------------------------
    def identified_dxdt(x_val, t_val):
        val = 0.0
        for coef, name in zip(beta_mean_refit, terms_significant):
            if name == "1":
                val += coef
            elif name.startswith("x^"):
                p = float(name.split("^")[1])
                val += coef * (x_val ** p)
            elif name.startswith("t^"):
                p = float(name.split("^")[1])
                val += coef * (t_val ** p)
            elif name.startswith("(x/t)^"):
                p = float(name.split("^")[1])
                val += coef * (0.0 if t_val == 0 else (x_val / t_val) ** p)
            else:
                # 若添加更多项类型，可在此扩展
                pass
        return val

    # -----------------------------
    # RK4积分（含外推）
    # -----------------------------
    t_extended = np.concatenate([np.array(t), np.array(t[-1] + dt * np.arange(1, h+1))]) if h>0 else np.array(t)
    x_identified = [float(x[0])]
    for i in range(len(t_extended)-1):
        ti = float(t_extended[i])
        xi = float(x_identified[i])
        k1 = identified_dxdt(xi, ti)
        k2 = identified_dxdt(xi + 0.5*dt*k1, ti + 0.5*dt)
        k3 = identified_dxdt(xi + 0.5*dt*k2, ti + 0.5*dt)
        k4 = identified_dxdt(xi + dt*k3, ti + dt)
        x_next = xi + dt*(k1 + 2*k2 + 2*k3 + k4)/6
        x_identified.append(x_next)
    x_identified = np.array(x_identified)

    # -----------------------------
    # 返回结果
    # -----------------------------
    return {
        "t": t_extended,
        "x_true": np.concatenate([np.array(x), np.array([np.nan]*h)]) if h>0 else np.array(x),
        "x_identified": x_identified,
        "beta_mean": beta_mean_refit,
        "beta_samples": beta_samples_refit,
        "beta_low": beta_q025_refit,
        "beta_high": beta_q975_refit,
        "identified_formula": identified_formula,
        "terms_significant": terms_significant,
        "selected_mask": significant_mask,
        "w_mean_all": w_mean,
        "w_q025_all": w_q025,
        "w_q975_all": w_q975
    }

# ==============================
# 主程序示例（保持你原来的动态）
# ==============================
if __name__ == "__main__":
    # 生成示例数据（注意 t 从 dt 开始避免 t==0）
    T = 10.0
    dt = 0.02
    t = np.linspace(dt, T, int(T/dt))  # 从 dt 开始
    x0 = 0.5
    x = [x0]
    for idx in range(1, len(t)):
        # 真实动力学： dx/dt = 1.2 x - 1.3 x^3 + 3 * (x/t)^0.4
        safe_xt = 0.0 if t[idx] == 0 else (x[-1] / t[idx])**0.4
        x_next = x[-1] + dt*(1.2 * x[-1] - 1.3 * x[-1]**3 + 3.0 * safe_xt)
        x.append(x_next)
    x = np.array(x)

    # 拟合 + 外推 50 步
    results = bayesian_sindy_fit_refit(x, t, dt=dt, h=50, seed=0,
                                      num_warmup=600, num_samples=1200,
                                      coef_threshold=0.05, max_power=3)

    # 输出辨识公式
    print("\n=====================================")
    print(f"辨识公式: dx/dt = {results['identified_formula']}")
    print("=====================================\n")

    # 输出第一步（ARD）所有 w 的后验均值与 CI（便于诊断）
    print("---- ARD first-step posterior (all features) ----")
    for name, mean, lo, hi, sel in zip(
            ["1"] + [f"x^{p}" for p in range(1,4)] + [f"t^{p}" for p in range(1,4)] + [f"(x/t)^{p}" for p in range(1,4)],
            results["w_mean_all"],
            results["w_q025_all"],
            results["w_q975_all"],
            results["selected_mask"]):
        print(f"{name:>8s} : mean={mean:.4f}, 95% CI=({lo:.4f}, {hi:.4f}), selected={int(sel)}")

    # 输出 refit 参数与 95% CI
    if len(results["terms_significant"])>0:
        print("\n各参数均值和 95% 置信区间（refit）:")
        print("-------------------------------------")
        for name, mean, lo, hi in zip(results['terms_significant'],
                                       results['beta_mean'],
                                       results['beta_low'],
                                       results['beta_high']):
            print(f"{name:>10s} : mean={mean:.4f}, 95% CI=({lo:.4f}, {hi:.4f})")
        print("-------------------------------------")
    else:
        print("没有显著项 (selected)，返回零模型。")

    # 绘图对比（原始数据 + 外推）
    plt.figure(figsize=(10,5))
    plt.plot(results['t'][:len(t)], x, label='真实 x(t)')
    plt.plot(results['t'], results['x_identified'], '--', label='辨识公式模拟 x(t) + 外推')
    plt.xlabel('t')
    plt.ylabel('x')
    plt.title('ARD 贝叶斯SINDy: 辨识公式预测 + 外推')
    plt.legend()
    plt.grid(True)
    plt.show()
