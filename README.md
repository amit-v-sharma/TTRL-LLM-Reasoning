# Test-Time Reinforcement Learning (TTRL) - Tutorial

Welcome to the **DeepLearning.AI TTRL Course**.

**Test-Time Reinforcement Learning (TTRL)** is the paradigm shift that enables LLMs to improve _during inference_ (test-time) rather than just during pre-training. This is the technology behind advanced reasoning models like **OpenAI's o1** and **Google's Strawberry**.

## 📚 Syllabus

1.  **Lesson 1: The Intuition of TTRL**
    - What is "Test-Time scaling"?
    - Simulating Consensus (Majority Voting).
    - The "Pseudo-Reward" mechanism.
    - _Code:_ `01_Intro_to_TTRL.py`

2.  **Lesson 2: Search & Verification (Coming Soon)**
    - Implementing "Best-of-N" search.
    - Building a lightweight Verifier model.

3.  **Lesson 3: Online Policy Optimization (Advanced)**
    - Updating the model context (In-Context TTRL).
    - PPO/GRPO loops at inference time.

## 🚀 Getting Started

1.  **Install Dependencies**:

    ```bash
    pip install -r requirements.txt
    ```

2.  **Run Lesson 1**:
    ```bash
    python 01_Intro_to_TTRL.py
    ```

## 🧠 Core Concepts

- **Inference Compute**: Spending more time _thinking_ (generating tokens) to reduce errors.
- **Self-Correction**: The model evaluating its own outputs to find the best path.
- **Consensus**: Using the "wisdom of the crowd" (sampled from the same model) to find truth.
