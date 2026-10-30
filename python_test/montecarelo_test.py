"""
몬테카를로(Monte Carlo) 방법 직관적 시각화
------------------------------------------
아이디어:
  1) 한 변의 길이가 2인 정사각형(-1 ~ 1, -1 ~ 1) 안에 점을 무작위로 N개 뿌린다.
  2) 정사각형에 내접하는 반지름 1인 원을 그린다.
  3) 원 안에 떨어진 점의 개수를 세서, 그 비율로 원의 넓이를 추정한다.
     (원 넓이) / (정사각형 넓이) = π*r^2 / (2r)^2 = π/4
     => π ≈ 4 * (원 안에 떨어진 점 개수 / 전체 점 개수)

  점의 개수 N이 커질수록 추정값이 실제 π(3.14159...)에 점점 가까워지는
  "대수의 법칙(law of large numbers)"을 눈으로 확인할 수 있다.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 한글 폰트 설정 (나눔고딕) - 시스템마다 설치 경로가 달라서 여러 후보를 탐색
import os

# 이미지 저장 폴더: 이 스크립트 파일이 있는 폴더 (환경이 바뀌어도 항상 유효)
SAVE_DIR = os.path.dirname(os.path.abspath(__file__))

CANDIDATE_FONT_PATHS = [
    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",          # Ubuntu/Debian (apt install fonts-nanum)
    "/usr/share/fonts/nanum/NanumGothic.ttf",                    # 일부 배포판
    os.path.expanduser("~/.fonts/NanumGothic.ttf"),              # 사용자 로컬 설치
    "C:/Windows/Fonts/NanumGothic.ttf",                          # Windows
    "/Library/Fonts/NanumGothic.ttf",                            # macOS
]

font_path = next((p for p in CANDIDATE_FONT_PATHS if os.path.isfile(p)), None)

if font_path:
    fm.fontManager.addfont(font_path)
    plt.rcParams["font.family"] = fm.FontProperties(fname=font_path).get_name()
else:
    print("[경고] NanumGothic 폰트를 찾지 못했습니다. 한글 라벨이 깨져 보일 수 있습니다.")
    print("       Ubuntu/Debian: sudo apt install -y fonts-nanum && fc-cache -f -v")
    print("       macOS: brew install --cask font-nanum-gothic")

plt.rcParams["axes.unicode_minus"] = False  # 마이너스 기호 깨짐 방지

# 재현 가능한 결과를 위해 시드 고정 (원하면 제거하면 매번 다른 결과)
rng = np.random.default_rng(seed=42)

# ------------------------------------------------------------------
# 1. 점 뿌리기 (한 번에 최대 개수만큼 생성해두고, 앞에서부터 잘라서 사용)
# ------------------------------------------------------------------
# 콘솔 표에 출력할 N 값들 (여기에 값을 추가/변경하면 됨)
TABLE_SAMPLE_SIZES = [10, 100, 1000, 5000, 20000, 100000]

# N_MAX는 항상 TABLE_SAMPLE_SIZES 중 가장 큰 값 이상이어야 한다.
# (예전 버그: N_MAX를 20000으로 고정해두고 표에는 100000을 요청하면,
#  inside[:100000]이 넘파이 슬라이싱 특성상 에러 없이 있는 20000개만 반환해서
#  분모(n=100000)만 커지고 분자는 그대로라 추정값이 조용히 틀어졌음.
#  아래처럼 자동 계산 + assert로 이런 실수를 원천 차단한다.)
N_MAX = max(20000, max(TABLE_SAMPLE_SIZES))

x = rng.uniform(-1, 1, N_MAX)
y = rng.uniform(-1, 1, N_MAX)
inside = (x**2 + y**2) <= 1.0  # 원 안에 있는지 여부 (반지름 1인 원)

# ------------------------------------------------------------------
# 2. 그림 1: 점 뿌리기 산점도 (원 안 / 원 밖 색으로 구분)
#    -> "몬테카를로가 실제로 뭘 하는지" 직관적으로 보여줌
# ------------------------------------------------------------------
N_SCATTER = 3000  # 산점도는 너무 많으면 지저분하니 3000개만 표시

fig1, ax1 = plt.subplots(figsize=(6, 6))
ax1.scatter(x[:N_SCATTER][inside[:N_SCATTER]], y[:N_SCATTER][inside[:N_SCATTER]],
            s=4, color="#2563eb", label="원 안 (inside)")
ax1.scatter(x[:N_SCATTER][~inside[:N_SCATTER]], y[:N_SCATTER][~inside[:N_SCATTER]],
            s=4, color="#f97316", label="원 밖 (outside)")

# 원 경계선 그리기
theta = np.linspace(0, 2 * np.pi, 400)
ax1.plot(np.cos(theta), np.sin(theta), color="black", linewidth=1.5)

ax1.set_xlim(-1, 1)
ax1.set_ylim(-1, 1)
ax1.set_aspect("equal")
pi_est_scatter = 4 * inside[:N_SCATTER].sum() / N_SCATTER
ax1.set_title(f"몬테카를로 점 {N_SCATTER}개 뿌리기\n"
              f"원 안 비율로 추정한 π ≈ {pi_est_scatter:.4f}")
ax1.legend(loc="upper right", fontsize=8)
fig1.tight_layout()
fig1.savefig(os.path.join(SAVE_DIR, "monte_carlo_scatter.png"), dpi=150)

# ------------------------------------------------------------------
# 3. 그림 2: 점의 개수(N)를 늘려가며 π 추정값이 수렴하는 과정
#    -> "표본이 많아질수록 정확해진다"는 몬테카를로의 핵심 원리를 보여줌
# ------------------------------------------------------------------
sample_sizes = np.unique(np.logspace(1, np.log10(N_MAX), 200).astype(int))
cum_inside = np.cumsum(inside)  # N개까지 누적으로 원 안에 들어간 점 개수
pi_estimates = 4 * cum_inside[sample_sizes - 1] / sample_sizes

fig2, ax2 = plt.subplots(figsize=(7, 4.5))
ax2.plot(sample_sizes, pi_estimates, color="#2563eb", linewidth=1.2,
         label="몬테카를로 추정값")
ax2.axhline(np.pi, color="#dc2626", linestyle="--", linewidth=1.2,
            label=f"실제 π = {np.pi:.5f}")
ax2.set_xscale("log")
ax2.set_xlabel("점의 개수 N (로그 스케일)")
ax2.set_ylabel("π 추정값")
ax2.set_title("표본 개수(N)가 늘어남에 따른 π 추정값의 수렴")
ax2.legend()
ax2.grid(alpha=0.3)
fig2.tight_layout()
fig2.savefig(os.path.join(SAVE_DIR, "monte_carlo_convergence.png"), dpi=150)

# ------------------------------------------------------------------
# 4. 콘솔 출력: N을 늘려가며 추정 정확도가 좋아지는 걸 숫자로도 확인
# ------------------------------------------------------------------
print(f"{'N':>10} | {'π 추정값':>10} | {'실제 π와 오차':>12}")
print("-" * 40)
for n in TABLE_SAMPLE_SIZES:
    # 안전장치: n이 실제 생성된 점 개수(N_MAX)를 넘으면 조용히 틀린 값을 내는 대신
    # 여기서 바로 명확한 에러를 낸다.
    assert n <= N_MAX, f"n={n}이 N_MAX={N_MAX}보다 큽니다. TABLE_SAMPLE_SIZES를 늘렸다면 N_MAX도 같이 늘어나야 합니다."
    est = 4 * inside[:n].sum() / n
    print(f"{n:>10} | {est:>10.5f} | {abs(est - np.pi):>12.5f}")