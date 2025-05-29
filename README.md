# Reinforcement-Learning
# 강화학습 기초 - DQN 구현 (CartPole-v1 환경)

이 문서는 강화학습(Reinforcement Learning, RL)의 핵심 알고리즘 중 하나인 DQN(Deep Q-Network)의 이론적 배경을 설명하고, `gymnasium`의 `CartPole-v1` 환경에서 `Keras/TensorFlow`를 사용하여 DQN을 구현하는 과정을 안내합니다.

---

## 1. 강화학습 (Reinforcement Learning) 기본 개념

강화학습은 에이전트(Agent)가 환경(Environment)과 상호작용하며 누적 보상(Cumulative Reward)을 최대화하는 행동 정책(Policy)을 학습하는 머신러닝의 한 분야입니다.

* **에이전트(Agent)**: 학습의 주체로, 환경의 상태(State)를 관찰하고 행동(Action)을 결정합니다.
* **환경(Environment)**: 에이전트 외부의 세계로, 에이전트의 행동에 따라 상태가 변하고 보상(Reward)을 제공합니다.
* **상태(State, $s$)**: 특정 시점에서 환경을 나타내는 정보입니다.
* **행동(Action, $a$)**: 에이전트가 특정 상태에서 취할 수 있는 결정입니다.
* **보상(Reward, $r$)**: 에이전트가 특정 상태에서 특정 행동을 취한 결과로 환경으로부터 받는 즉각적인 피드백입니다.
* **정책(Policy, $\pi$)**: 상태 $s$에서 행동 $a$를 선택할 확률 또는 결정론적 규칙 $\pi(a|s)$입니다. 강화학습의 목표는 누적 보상을 최대화하는 최적 정책 $\pi^*$를 찾는 것입니다.
* **에피소드(Episode)**: 시작 상태부터 종료 상태까지의 일련의 상호작용 시퀀스입니다.

강화학습 문제는 주로 **마르코프 결정 과정(Markov Decision Process, MDP)**으로 정형화됩니다. MDP는 (상태 집합 $S$, 행동 집합 $A$, 상태 전이 확률 $P(s'|s,a)$, 보상 함수 $R(s,a,s')$, 할인율 $\gamma$)로 구성됩니다.

---

## 2. Q-러닝 (Q-Learning)

Q-러닝은 모델 없이(model-free) 환경에 대한 사전 지식 없이 학습할 수 있는 대표적인 가치 기반(value-based) 강화학습 알고리즘입니다.

* **행동-가치 함수 (Action-Value Function, Q-function, $Q(s, a)$)**:
    상태 $s$에서 행동 $a$를 취하고 그 이후 특정 정책 $\pi$를 따랐을 때 얻을 수 있는 미래 누적 보상의 기댓값을 의미합니다.
  $Q^{\pi}(s,a) = E_{\pi}[R\_{t+1} + \gamma R\_{t+2} + \gamma^2 R\_{t+3} + \ldots | S\_t=s, A\_t=a]$

    여기서 $\gamma$는 할인율(discount factor, $0 \le \gamma \le 1$)로, 현재 보상 대비 미래 보상의 가치를 얼마나 크게 볼 것인지를 결정합니다.

* **벨만 최적 방정식 (Bellman Optimality Equation for Q-function)**:
    최적의 행동-가치 함수 $Q(s,a)$는 모든 상태 $s$와 행동 $a$에 대해 다음 벨만 방정식을 만족합니다. 이는 현재 상태 $s$에서 행동 $a$를 취했을 때의 즉각적인 보상 $r$과, 다음 상태 $s'$에서 가능한 모든 행동 $a'$ 중 가장 큰 Q-값을 선택했을 때의 할인된 미래 가치의 합으로 표현됩니다.
    <p align="center">
    <img src="https://latex.codecogs.com/png.latex?\dpi{120}&space;Q^*(s,a)&space;=&space;\mathbb{E}\bigl[R_{t+1}&space;+&space;\gamma\max_{a'}Q^*(s_{t+1},a')\mid&space;s,a\bigr]" alt="Bellman Optimality Equation" />
  </p>  
    일반적으로 상태 전이와 보상이 결정론적(deterministic)이라고 가정하면, 다음과 같이 단순화할 수 있습니다.
    <p align="center">
    <img src="https://latex.codecogs.com/png.latex?\dpi{120}&space;Q^*(s,a)&space;=&space;r&space;+&space;\gamma\max_{a'}Q^*(s',a')" alt="Simplified Bellman Equation" />
  </p>
  
* **Q-러닝 업데이트 규칙 (시간차 학습, Temporal Difference Learning)**:
    Q-러닝은 시간차 학습을 통해 $Q^*(s,a)$를 반복적으로 추정합니다. 현재의 Q 값 $Q(s,a)$와 시간차 목표(TD Target) $r + \gamma \max_{a'} Q(s',a')$ 사이의 오차를 줄이는 방향으로 Q 값을 업데이트합니다.
    <p align="center">
  <img src="https://latex.codecogs.com/png.latex?\dpi{120}&space;Q(s,a)&space;\leftarrow&space;Q(s,a)&space;&plus;&space;\alpha\bigl[r&space;&plus;&space;\gamma\max_{a'}Q(s',a')&space;-&space;Q(s,a)\bigr]" alt="Q-learning update" />
</p>
    여기서 $\alpha$는 학습률(learning rate)입니다.
---

## 3. 심층 Q-네트워크 (Deep Q-Network, DQN)

상태 공간이나 행동 공간이 매우 크거나 연속적인 경우, 모든 $(s,a)$ 쌍에 대한 Q-값을 테이블 형태로 저장하는 것은 불가능합니다. DQN은 이러한 한계를 극복하기 위해 심층 신경망(Deep Neural Network)을 사용하여 Q-함수를 근사합니다. 이를 $Q(s, a; \theta) \approx Q^*(s,a)$로 표현하며, $\theta$는 신경망의 가중치(파라미터)입니다.

* **네트워크 아키텍처**:
    일반적으로 DQN은 상태 $s$를 입력으로 받아, 가능한 모든 행동 $a$에 대한 Q-값 벡터를 출력하는 신경망 구조를 가집니다.
    제공된 코드에서는 `CartPole-v1` 환경의 상태(4차원 벡터)를 입력으로 받아, 두 개의 은닉층(각각 24개의 뉴런, ReLU 활성화 함수)을 거쳐, 출력층에서 각 행동(왼쪽/오른쪽, 2개)에 대한 Q-값을 선형(linear) 활성화 함수로 출력합니다.
    * 입력층: 상태 $s$ (예: `state_size=4`)
    * 은닉층 1: `Dense(24, activation='relu')`
    * 은닉층 2: `Dense(24, activation='relu')`
    * 출력층: `Dense(action_size, activation='linear')` (예: `action_size=2`)

* **손실 함수 (Loss Function)**:
    DQN은 Q-러닝의 업데이트 목표와 신경망의 예측값 간의 오차를 최소화하도록 학습됩니다. 주로 평균 제곱 오차(Mean Squared Error, MSE)를 손실 함수로 사용합니다.
    * 타겟 Q-값 (TD Target)**:
      <p align="center">
    <img src="https://latex.codecogs.com/png.latex?\dpi{120}&space;y_j&space;=&space;\begin{cases}
      r_j, & \text{if episode terminates at step }\,j+1\\
      r_j&space;+&space;\gamma\max_{a'}Q(s'_j,a';\theta^-),&\text{otherwise}
    \end{cases}" alt="TD Target" />
  </p>  
      (여기서 $\theta^-$는 타겟 네트워크의 파라미터입니다. 아래 설명 참조)
    * 손실 함수: <p align="center">
    <img src="https://latex.codecogs.com/png.latex?\dpi{120}&space;L(\theta)&space;=&space;\mathbb{E}_{(s,a,r,s')\sim\mathcal{U}(D)}\bigl[\bigl(y_j&space;-&space;Q(s_j,a_j;\theta)\bigr)^2\bigr]" alt="DQN Loss" />
  </p> 
        ($U(D)$는 경험 리플레이 버퍼 $D$에서 샘플링된 경험 분포)

* **옵티마이저 (Optimizer)**:
    계산된 손실을 바탕으로 신경망의 가중치 $\theta$를 업데이트하기 위해 옵티마이저를 사용합니다. 코드에서는 `Adam` 옵티마이저를 사용합니다.

---

## 4. DQN의 핵심 기법

DQN은 안정적이고 효율적인 학습을 위해 몇 가지 핵심 기법을 사용합니다.

* **경험 재사용 (Experience Replay)**:
    * **개념**: 에이전트가 환경과 상호작용하면서 얻는 경험(transition: $(s_t, a_t, r_{t+1}, s_{t+1}, \text{done})$)을 리플레이 버퍼(replay buffer) 또는 메모리(memory)라는 유한한 크기의 저장 공간에 순차적으로 저장합니다. 학습 시에는 이 버퍼에서 미니배치(minibatch)만큼의 경험을 무작위로 샘플링하여 신경망 업데이트에 사용합니다.
    * **목적 및 장점**:
        1.  **데이터 효율성 증대**: 하나의 경험이 여러 번의 학습에 재사용될 수 있어 데이터 효율성을 높입니다.
        2.  **학습 데이터 간 상관관계 감소**: 시간 순서대로 경험을 학습하면 데이터 간 상관관계가 높아 학습이 불안정해질 수 있습니다. 무작위 샘플링은 이러한 상관관계를 깨뜨려 학습을 안정화시킵니다.
        3.  **학습의 평활화**: 다양한 시점의 경험들을 섞어 학습함으로써 특정 시점의 경험에 과도하게 편향되는 것을 방지하고 학습 과정을 부드럽게 만듭니다.
    * **구현**: `ReplayBuffer` 클래스로 구현되며, `add` 메서드로 경험을 추가하고 `sample` 메서드로 미니배치를 추출합니다. 버퍼가 가득 차면 가장 오래된 경험부터 삭제합니다 (FIFO 방식).

* **타겟 네트워크 (Target Network)**:
    * **개념**: Q-러닝 업데이트 시 사용되는 타겟 Q-값($r + \gamma \max_{a'} Q(s',a')$)을 계산할 때, 현재 업데이트되고 있는 주 네트워크($Q(s,a;\theta)$)와 동일한 네트워크를 사용하면 타겟 값이 계속 변하여 학습이 불안정해질 수 있습니다. (타겟이 예측값을 따라 흔들리는 문제)
        이를 해결하기 위해, 주 네트워크와 동일한 구조를 가지지만 파라미터($\theta^-$)가 주기적으로만 동기화되는 별도의 '타겟 네트워크' $Q(s,a;\theta^-)$를 사용합니다.
    * **동작 방식**:
        1.  초기에는 주 네트워크의 파라미터 $\theta$를 타겟 네트워크의 파라미터 $\theta^-$로 복사합니다.
        2.  학습 과정에서 주 네트워크의 파라미터 $\theta$는 매 스텝 업데이트되지만, 타겟 네트워크의 파라미터 $\theta^-$는 고정되어 있습니다.
        3.  일정 주기(예: `target_update_freq` 스텝 또는 에피소드마다)마다 주 네트워크의 최신 파라미터 $\theta$를 타겟 네트워크의 파라미터 $\theta^-$로 복사하여 업데이트합니다.
    * **목적 및 장점**: 타겟 Q-값을 좀 더 안정적으로 고정시켜줌으로써 학습의 진동을 줄이고 수렴을 돕습니다.

* **입실론-그리디 (Epsilon-Greedy, ε-greedy) 탐색**:
    * **개념**: 강화학습에서 에이전트는 현재까지 학습한 지식을 최대한 활용(Exploitation)하는 것과, 더 나은 보상을 얻을 수 있는 새로운 행동을 시도(Exploration)하는 것 사이의 균형을 맞춰야 합니다. 이를 탐색-활용 딜레마(Exploration-Exploitation Dilemma)라고 합니다.
    * **동작 방식**:
        * 확률 $\epsilon$ (입실론, $0 \le \epsilon \le 1$) 만큼은 무작위로 행동을 선택하여 탐색합니다.
        * 확률 $1-\epsilon$ 만큼은 현재 Q-함수 예측에 따라 가장 높은 Q-값을 주는 행동을 선택하여 활용합니다.
    * **입실론 감쇠 (Epsilon Decay)**: 학습 초기에는 $\epsilon$ 값을 높게 설정하여(예: 1.0) 다양한 탐색을 장려하고, 학습이 진행됨에 따라 $\epsilon$ 값을 점진적으로 감소시켜(예: 매 에피소드마다 `epsilon_decay` 비율로 곱함) 활용의 비중을 높입니다. $\epsilon$ 값은 최소값(`epsilon_min`) 이하로 내려가지 않도록 설정하여 최소한의 탐색을 보장합니다.
    * **구현**: `select_action` 함수에서 `np.random.rand() < epsilon` 조건으로 탐색 또는 활용을 결정합니다.

---

## 5. Gymnasium (Gym) 환경: CartPole-v1

* **Gymnasium**: 강화학습 알고리즘을 개발하고 비교하기 위한 표준화된 환경 모음(toolkit)입니다. (OpenAI Gym의 후속 버전)
* **CartPole-v1 환경**:
    * **목표**: 카트(cart) 위에 수직으로 세워진 막대기(pole)가 쓰러지지 않도록 카트를 왼쪽 또는 오른쪽으로 움직여 균형을 유지하는 것입니다.
    * **상태 공간 (State Space)**: 연속적인 4개의 값으로 구성됩니다.
        1.  카트 위치 (Cart Position)
        2.  카트 속도 (Cart Velocity)
        3.  막대기 각도 (Pole Angle)
        4.  막대기 끝 속도 (Pole Velocity At Tip)
        * `state_size = env.observation_space.shape[0]`는 4가 됩니다.
    * **행동 공간 (Action Space)**: 이산적인 2개의 행동으로 구성됩니다.
        1.  0: 카트를 왼쪽으로 밀기
        2.  1: 카트를 오른쪽으로 밀기
        * `action_size = env.action_space.n`은 2가 됩니다.
    * **보상 (Reward)**: 매 타임스텝마다 막대기가 쓰러지지 않고 균형을 유지하면 +1의 보상을 받습니다.
    * **종료 조건 (Termination)**:
        1.  막대기 각도가 특정 한계(예: ±12도)를 벗어나는 경우
        2.  카트 위치가 특정 한계(예: ±2.4)를 벗어나는 경우
        3.  에피소드 길이가 특정 최대치(예: CartPole-v1에서는 주로 200 또는 500 타임스텝)에 도달하는 경우 (Truncation)

---

## 6. DQN 학습 루프

제공된 코드의 전체적인 DQN 학습 과정은 다음과 같습니다.

1.  **초기화**:
    * 하이퍼파라미터 설정 (학습률, 할인율, 입실론 값, 버퍼 크기, 배치 크기 등).
    * 주 네트워크($Q(s,a;\theta)$)와 타겟 네트워크($Q(s,a;\theta^-)$) 생성. 초기에는 두 네트워크의 가중치를 동일하게 설정.
    * 경험 리플레이 버퍼 생성.

2. **에피소드 반복 (총 `num_episodes` 만큼)**  
   a. **환경 초기화**  
      - 에피소드 시작 시 `state = env.reset()` 으로 초기 상태 \(s_0\) 를 받습니다.  
      - `total_reward = 0` 으로 초기화합니다.  
   b. **타임스텝 반복** (에피소드 내 최대 스텝 수 또는 `done==True` 될 때까지)  
      1. **행동 선택**  
         ```python
         if np.random.rand() < epsilon:
             action = env.action_space.sample()   # 탐색
         else:
             action = np.argmax(model.predict(state))  # 활용
         ```  
      2. **행동 수행**  
         ```python
         next_state, reward, done, info = env.step(action)
         ```  
         → 다음 상태 \(s_{t+1}\), 보상 \(r_{t+1}\), 종료 여부 `done` 등을 반환  
      3. **경험 저장**  
         ```python
         replay_buffer.add((state, action, reward, next_state, done))
         ```  
      4. **상태·보상 업데이트**  
         ```python
         state = next_state
         total_reward += reward
         ```  
      5. **종료 확인**  
         - `if done: break`  
   c. **모델 학습** (`train_model` 호출)  
      - 리플레이 버퍼에서 무작위로 `batch_size` 개의 경험을 샘플링  
      - 각 경험에 대해  
        - 타겟 네트워크로 타겟 Q값 \(y_j\) 계산  
        - 주 네트워크로 예측 Q값 \(Q(s_j,a_j;\theta)\) 계산  
      - MSE 손실  
        $$
          L(\theta)
          = \mathbb{E}\bigl[(y_j - Q(s_j,a_j;\theta))^2\bigr]
        $$  
        를 최소화하도록 한 배치 학습  
   d. **타겟 네트워크 업데이트**  
      - 매 `target_update_freq` 에피소드마다  
        ```python
        target_model.set_weights(model.get_weights())
        ```  
      - 파라미터 동기화 (\(\theta^- \leftarrow \theta\))  
   e. **입실론 값 감쇠**  
      ```python
      epsilon = max(epsilon_min, epsilon * epsilon_decay)
      ```  
      * **입실론 값 감쇠**  
  <p align="center">
    <img src="https://latex.codecogs.com/png.latex?\dpi{120}&space;\epsilon&space;=&space;\max\bigl(\epsilon_{\min},\;\epsilon\cdot\epsilon_{\text{decay}}\bigr)" alt="epsilon update" />
  </p>

    f.  **결과 출력**: 현재 에피소드 번호, 총 보상, 현재 입실론 값 등을 출력하여 학습 진행 상황을 모니터링합니다.
3.  **학습 완료 후**: 학습된 모델을 사용하여 실제 환경에서 에이전트가 어떻게 행동하는지 테스트하고 시각화할 수 있습니다. (코드의 마지막 `rewards` 리스트를 사용한 그래프 플로팅 및 학습된 모델 테스트 부분)

---

## 7. 결론

이 노트북은 강화학습의 기본적인 아이디어부터 시작하여, Q-러닝의 원리를 이해하고, 이를 심층 신경망과 결합한 DQN 알고리즘을 단계별로 구현하는 과정을 보여줍니다. 특히 경험 재사용, 타겟 네트워크, 입실론-그리디 탐색과 같은 DQN의 핵심 구성 요소들의 이론적 배경과 실제 코드 구현을 통해 강화학습 에이전트가 어떻게 환경과의 상호작용을 통해 최적 정책을 학습해 나가는지를 이해하는 데 도움을 줍니다. CartPole-v1이라는 간단하면서도 표준적인 환경에서의 실험은 DQN의 작동 방식을 명확하게 보여주는 좋은 예시입니다.

---

## 7. 사용 방법

```bash
git clone https://github.com/YourUser/Reinforcement-Learning.git
cd Reinforcement-Learning
pip install -r requirements.txt
jupyter notebook 01.QN.ipynb
```
---

## License

This project is licensed under the **MIT License**.

