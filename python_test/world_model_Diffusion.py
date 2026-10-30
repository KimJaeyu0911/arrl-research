"""
Diffusion World Model - 최소 예제 (방향성 확인용)
================================================

[목적]
World model이란 s_t, a_t -> s_{t+1} 의 환경 dynamics를 학습하는 모델을 뜻함.
이 코드는 그 dynamics 예측 함수를 "diffusion model(DDPM)"로 구현한
가장 단순한 버전임. 실제 논문/서비스 수준 구현이 아니라,
"diffusion을 world model에 어떻게 적용하는지" 그 뼈대만 보여주는 튜토리얼 코드.

[왜 diffusion을 world model에 쓰는가]
- 일반 MLP/RNN으로 s_{t+1}을 직접 회귀(regression)하면, 다음 상태가
  여러 가능성(multi-modal)을 가질 때 평균값으로 뭉개버리는 문제가 있음.
  (예: 갈림길에서 왼쪽/오른쪽 둘 다 가능하면 회귀 모델은 중간값을 뱉어버림)
- Diffusion model은 p(s_{t+1} | s_t, a_t)라는 "분포"를 직접 학습하기 때문에
  이런 multi-modal한 미래를 표현할 수 있음. 이게 diffusion을 world model에
  쓰는 핵심 동기임.

[실제 연구 사례 - 확실한 출처 기반]
- DIAMOND (NeurIPS 2024, "Diffusion for World Modeling: Visual Details Matter
  in Atari", https://diamond-wm.github.io/): Atari 게임 화면(이미지) 자체를
  diffusion으로 예측하는 world model. 이 코드보다 훨씬 큰 U-Net 기반 구조 사용.
- Diffusion World Model (NeurIPS 2024, arXiv, offline RL): 여러 스텝을 한번에
  rollout하는 방식으로 diffusion을 사용.
- JEDI (2026, arXiv:2605.13013): latent space에서 diffusion으로 world model을
  학습하는 최신 연구 (온라인 model-based RL).
=> 위 세 건은 검색으로 확인한 논문/공식 페이지 기반이라 정확한 정보로 분류함.
   다만 이 코드는 그 논문들의 공식 구현이 아니라 개념 이해용 자체 작성 코드이며,
   실제 논문 코드와는 세부 구조가 다를 수 있음 (이 부분은 제가 임의로 단순화한
   것이므로 "논문 그대로의 구현"은 아님을 명시함).

[이 코드의 구성]
1. 아주 단순한 pendulum(진자) dynamics로 (s_t, a_t) -> s_{t+1} 데이터 생성
2. Denoiser 네트워크: (noisy s_{t+1}, timestep t, condition=(s_t,a_t)) -> noise 예측
3. DDPM 학습 (forward diffusion + noise 예측 loss)
4. Reverse diffusion으로 s_{t+1} 샘플링 (실제 다음 상태와 비교)

필요 라이브러리: torch, numpy (둘 다 설치됨)
"""

import numpy as np
import torch
import torch.nn as nn

torch.manual_seed(0)
np.random.seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"


# ------------------------------------------------------------
# 1. 토이 환경: 단순 진자(pendulum) dynamics
#    state = [theta, theta_dot], action = torque (scalar)
#    실제 로봇 환경 대신, world model 개념만 보기 위한 최소 예시임.
# ------------------------------------------------------------
DT = 0.05
G, L = 9.8, 1.0


def pendulum_step(state, action):
    theta, theta_dot = state[:, 0], state[:, 1]
    torque = action[:, 0]
    theta_ddot = -(G / L) * np.sin(theta) + torque
    new_theta_dot = theta_dot + theta_ddot * DT
    new_theta = theta + new_theta_dot * DT
    # 약간의 환경 노이즈 추가 (world model이 분포를 학습할 이유를 만들어줌)
    noise = np.random.normal(0, 0.01, size=new_theta.shape)
    return np.stack([new_theta + noise, new_theta_dot], axis=1)


def make_dataset(n=20000):
    theta = np.random.uniform(-np.pi, np.pi, size=(n, 1))
    theta_dot = np.random.uniform(-2, 2, size=(n, 1))
    action = np.random.uniform(-1, 1, size=(n, 1))
    state = np.concatenate([theta, theta_dot], axis=1)
    next_state = pendulum_step(state, action)
    return (
        torch.tensor(state, dtype=torch.float32),
        torch.tensor(action, dtype=torch.float32),
        torch.tensor(next_state, dtype=torch.float32),
    )


# ------------------------------------------------------------
# 2. DDPM 스케줄 (noise schedule)
#    참고: Ho et al., "Denoising Diffusion Probabilistic Models", 2020
#    (이 스케줄 정의 자체는 DDPM 원 논문에서 확립된 표준 방식임)
# ------------------------------------------------------------
T = 100  # diffusion step 수 (간단히 하려고 적게 설정함)
betas = torch.linspace(1e-4, 0.02, T).to(device)
alphas = (1.0 - betas).to(device)
alphas_cumprod = torch.cumprod(alphas, dim=0).to(device)
# -> t가 GPU 텐서일 때 alphas_cumprod[t] 같은 인덱싱이 device mismatch 없이 되도록
#    처음부터 같은 device에 올려둠 (에러: "indices should be either on cpu or on
#    the same device as the indexed tensor" 방지)


def q_sample(x0, t, noise):
    """forward diffusion: x0 -> x_t (노이즈를 씌우는 과정)"""
    sqrt_ac = alphas_cumprod[t].sqrt().unsqueeze(-1)
    sqrt_1m_ac = (1 - alphas_cumprod[t]).sqrt().unsqueeze(-1)
    return sqrt_ac * x0 + sqrt_1m_ac * noise


# ------------------------------------------------------------
# 3. Denoiser 네트워크
#    입력: noisy next_state(x_t), timestep t, condition(s_t, a_t)
#    출력: 예측한 noise (epsilon)
#    -> 이 네트워크가 "world model"의 실체임. 상태공간이 저차원(2차원)이라
#       이미지용 U-Net 대신 MLP로 충분함. 이미지/고차원 관측이라면
#       DIAMOND처럼 U-Net 구조가 필요함.
# ------------------------------------------------------------
class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(
            -np.log(10000) * torch.arange(half, device=t.device).float() / half
        )
        args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class DenoiserMLP(nn.Module):
    def __init__(self, state_dim=2, action_dim=1, time_dim=32, hidden=128):
        super().__init__()
        self.time_embed = SinusoidalTimeEmbedding(time_dim)
        in_dim = state_dim + state_dim + action_dim + time_dim  # x_t, s_t, a_t, t
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, state_dim),
        )

    def forward(self, x_t, t, s_t, a_t):
        t_emb = self.time_embed(t)
        h = torch.cat([x_t, s_t, a_t, t_emb], dim=-1)
        return self.net(h)


# ------------------------------------------------------------
# 4. 학습 루프
# ------------------------------------------------------------
def train():
    s, a, s_next = make_dataset()
    s, a, s_next = s.to(device), a.to(device), s_next.to(device)

    model = DenoiserMLP().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    n_epochs = 300
    batch_size = 256
    n_samples = s.shape[0]

    for epoch in range(n_epochs):
        perm = torch.randperm(n_samples)
        total_loss = 0.0
        for i in range(0, n_samples, batch_size):
            idx = perm[i:i + batch_size]
            s_b, a_b, x0_b = s[idx], a[idx], s_next[idx]

            t_b = torch.randint(0, T, (x0_b.shape[0],), device=device)
            noise = torch.randn_like(x0_b)
            x_t = q_sample(x0_b, t_b, noise)

            noise_pred = model(x_t, t_b, s_b, a_b)
            loss = ((noise_pred - noise) ** 2).mean()

            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * x0_b.shape[0]

        if (epoch + 1) % 50 == 0:
            print(f"epoch {epoch+1}/{n_epochs}  loss={total_loss/n_samples:.5f}")

    return model


# ------------------------------------------------------------
# 5. Reverse diffusion으로 s_{t+1} 샘플링
#    (DDPM 원 논문의 표준 reverse process 수식을 그대로 사용)
# ------------------------------------------------------------
@torch.no_grad()
def sample_next_state(model, s_t, a_t, n_samples=1):
    s_t = s_t.repeat(n_samples, 1).to(device)
    a_t = a_t.repeat(n_samples, 1).to(device)
    x = torch.randn(n_samples, 2, device=device)

    for t in reversed(range(T)):
        t_batch = torch.full((n_samples,), t, device=device, dtype=torch.long)
        beta_t = betas[t].to(device)
        alpha_t = alphas[t].to(device)
        alpha_bar_t = alphas_cumprod[t].to(device)

        noise_pred = model(x, t_batch, s_t, a_t)
        mean = (1 / alpha_t.sqrt()) * (
            x - (beta_t / (1 - alpha_bar_t).sqrt()) * noise_pred
        )
        if t > 0:
            z = torch.randn_like(x)
            x = mean + beta_t.sqrt() * z
        else:
            x = mean
    return x.cpu()


# ------------------------------------------------------------
# 6. 궤적(rollout) 비교: 실제 dynamics vs diffusion world model
#    world model을 여러 스텝 동안 "자기 자신의 예측을 다시 입력으로" 넣어서
#    (autoregressive rollout) 실제 환경 궤적과 비교함.
#    -> 이게 world model을 planning/제어에 실제로 쓸 때의 방식임.
#    주의: 스텝이 진행될수록 오차가 누적되는 "compounding error"가
#    diffusion world model을 포함한 모든 학습 기반 world model의
#    잘 알려진 한계임 (world model 관련 논문들이 공통적으로 언급하는 문제).
# ------------------------------------------------------------
@torch.no_grad()
def rollout_true(state0, actions):
    states = [state0]
    s = state0.copy()
    for a in actions:
        s = pendulum_step(s, a.reshape(1, -1))
        states.append(s)
    return np.concatenate(states, axis=0)  # (n_steps+1, 2)


@torch.no_grad()
def rollout_model(model, state0, actions, n_rollouts=5):
    # n_rollouts개의 독립적인 diffusion rollout을 동시에 진행 (불확실성 시각화용)
    s = torch.tensor(state0, dtype=torch.float32).repeat(n_rollouts, 1).to(device)
    all_states = [s.cpu().numpy()]
    for a in actions:
        a_t = torch.tensor(a, dtype=torch.float32).reshape(1, -1).repeat(n_rollouts, 1).to(device)
        # 배치 전체를 한번에 reverse diffusion (n_samples=1 호출 대신 배치로 처리)
        x = torch.randn(n_rollouts, 2, device=device)
        for t in reversed(range(T)):
            t_batch = torch.full((n_rollouts,), t, device=device, dtype=torch.long)
            beta_t, alpha_t, alpha_bar_t = betas[t], alphas[t], alphas_cumprod[t]
            noise_pred = model(x, t_batch, s, a_t)
            mean = (1 / alpha_t.sqrt()) * (x - (beta_t / (1 - alpha_bar_t).sqrt()) * noise_pred)
            if t > 0:
                x = mean + beta_t.sqrt() * torch.randn_like(x)
            else:
                x = mean
        s = x
        all_states.append(s.cpu().numpy())
    return np.stack(all_states, axis=1)  # (n_rollouts, n_steps+1, 2)


def plot_rollout_comparison(model, n_steps=40, n_rollouts=5):
    import matplotlib
    import matplotlib.font_manager as fm
    import matplotlib.pyplot as plt

    # 한글 폰트 설정 (시스템에 NanumGothic이 없어 Noto Sans CJK KR로 대체)
    kr_font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    fm.fontManager.addfont(kr_font_path)
    matplotlib.rcParams["font.family"] = fm.FontProperties(fname=kr_font_path).get_name()
    matplotlib.rcParams["axes.unicode_minus"] = False

    state0 = np.array([[0.5, -0.3]], dtype=np.float32)
    actions = np.random.uniform(-0.3, 0.3, size=(n_steps, 1)).astype(np.float32)

    true_traj = rollout_true(state0, actions)              # (n_steps+1, 2)
    model_traj = rollout_model(model, state0, actions, n_rollouts)  # (n_rollouts, n_steps+1, 2)

    t_axis = np.arange(n_steps + 1) * DT

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # (a) theta(각도) 시간에 따른 변화: 실제 vs diffusion rollout들
    ax = axes[0]
    for i in range(n_rollouts):
        ax.plot(t_axis, model_traj[i, :, 0], color="tab:orange", alpha=0.4,
                label="diffusion rollout" if i == 0 else None)
    ax.plot(t_axis, true_traj[:, 0], color="tab:blue", linewidth=2.5, label="실제(true) dynamics")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("theta (rad)")
    ax.set_title("각도 궤적 비교: 실제 vs diffusion world model rollout")
    ax.legend()
    ax.grid(alpha=0.3)

    # (b) phase plot: theta vs theta_dot
    ax = axes[1]
    for i in range(n_rollouts):
        ax.plot(model_traj[i, :, 0], model_traj[i, :, 1], color="tab:orange", alpha=0.4,
                label="diffusion rollout" if i == 0 else None)
    ax.plot(true_traj[:, 0], true_traj[:, 1], color="tab:blue", linewidth=2.5, label="실제(true) dynamics")
    ax.scatter(*state0[0], color="black", zorder=5, label="시작점 (s_0)")
    ax.set_xlabel("theta (rad)")
    ax.set_ylabel("theta_dot (rad/s)")
    ax.set_title("Phase plot: 실제 vs diffusion world model rollout")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig("rollout_comparison.png", dpi=150)
    print("\n그래프 저장됨: rollout_comparison.png")


# ------------------------------------------------------------
# 7. 실행: 학습 + 실제 dynamics와 비교
# ------------------------------------------------------------
if __name__ == "__main__":
    model = train()

    # 테스트: 임의의 (s_t, a_t)에 대해 실제 다음 상태 vs diffusion 예측 비교
    test_state = torch.tensor([[0.5, -0.3]], dtype=torch.float32)
    test_action = torch.tensor([[0.2]], dtype=torch.float32)

    true_next = pendulum_step(test_state.numpy(), test_action.numpy())
    pred_next = sample_next_state(model, test_state, test_action, n_samples=5)

    print("\n[비교] s_t=", test_state.numpy(), " a_t=", test_action.numpy())
    print("실제 dynamics로 계산한 s_(t+1):", true_next)
    print("diffusion world model이 샘플링한 s_(t+1) (5개 샘플):")
    print(pred_next.numpy())

    # 여러 스텝 rollout 비교 그래프 생성 (rollout_comparison.png로 저장됨)
    plot_rollout_comparison(model, n_steps=40, n_rollouts=5)