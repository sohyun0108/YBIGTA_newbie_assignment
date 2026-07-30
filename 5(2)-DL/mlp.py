import time
import numpy as np


class MultiLayerPerceptron(object):
    def __init__(
        self,
        nn_input_dim,
        nn_hdim1,
        nn_hdim2,
        nn_hdim3,
        nn_output_dim,
        init="random",
        seed=0,
    ):

        """
        Descriptions:
            W1: First layer weights
            b1: First layer biases
            W2: Second layer weights
            b2: Second layer biases
            W3: Third layer weights
            b3: Third layer biases
            W4: Fourth layer weights
            b4: Fourth layer biases

        Args:
            nn_input_dim: input dimension
            nn_hdim1: number of neurons in hidden layer 1
            nn_hdim2: number of neurons in hidden layer 2
            nn_hdim3: number of neurons in hidden layer 3
            nn_output_dim: output dimension
            init: {'random', 'constant'}
            seed: random seed
        
        Returns:
        """
        # reset seed before start
        np.random.seed(seed)
        self.model = {}

        if init == "random":
            self.model["W1"] = np.random.randn(nn_input_dim, nn_hdim1)
            self.model["b1"] = np.zeros((1, nn_hdim1))
            self.model["W2"] = np.random.randn(nn_hdim1, nn_hdim2)
            self.model["b2"] = np.zeros((1, nn_hdim2))
            self.model["W3"] = np.random.randn(nn_hdim2, nn_hdim3)
            self.model["b3"] = np.zeros((1, nn_hdim3))
            self.model["W4"] = np.random.randn(nn_hdim3, nn_output_dim)
            self.model["b4"] = np.zeros((1, nn_output_dim))

        elif init == "constant":
            self.model["W1"] = np.ones((nn_input_dim, nn_hdim1))
            self.model["b1"] = np.zeros((1, nn_hdim1))
            self.model["W2"] = np.ones((nn_hdim1, nn_hdim2))
            self.model["b2"] = np.zeros((1, nn_hdim2))
            self.model["W3"] = np.ones((nn_hdim2, nn_hdim3))
            self.model["b3"] = np.zeros((1, nn_hdim3))
            self.model["W4"] = np.ones((nn_hdim3, nn_output_dim))
            self.model["b4"] = np.zeros((1, nn_output_dim))
        else:
            raise ValueError("init must be 'random' or 'constant'")

    def forward_propagation(self, X):
        """
        Forward pass of the network to compute the hidden layer features and classification scores. 
                
        Args:
            X: Input data of shape (N, D)
                    
        Returns:
            y_hat: (numpy array) Array of shape (N,) giving the classification scores for X
            cache: (dict) Values needed to compute gradients
        """
        W1, b1 = self.model["W1"], self.model["b1"]
        W2, b2 = self.model["W2"], self.model["b2"]
        W3, b3 = self.model["W3"], self.model["b3"]
        W4, b4 = self.model["W4"], self.model["b4"]

        ### CODE HERE ###
        # 행렬 연산 후 activation function 적용
        # y_hat을 X.shape[0] 크기로 reshape해야한다.
        # 1번째 은닉층: 선형 결합(X * W1 + b1) 후 ReLU 활성화 함수 적용
        # NumPy에서 행렬 곱은 @ 기호를 사용 (딥러닝에서는 입력 데이터 행렬과 가중치 행렬을 곱할 때 행렬 곱을 사용!)
        h1 = X @ W1 + b1 # X @ W1의 결과는 (데이터 수, 10) 크기의 행렬이 되는데 여기에 (1,10) 크기의 편향 b1을 더해주면 Numpy가 자동으로 데이터 N개 각각에 편향을 더해준다. (브로드캐스팅)
        z1 = relu(h1)

        # 2번째 은닉층: 1층의 출력(z1)을 입력받아 LeakyReLU 활성화 함수 적용
        h2 = z1 @ W2 + b2
        z2 = leakyrelu(h2)

        # 3번째 은닉층: 2층의 출력(z2)을 입력받음
        # 이전 층의 정보를 뛰어넘어 전달하는 이 기법을 Skip Connection(Residual Connection)이라고 부른다.
        h3 = z2 @ W3 + b3
        z3 = tanh(h3 + h1) #명세서 공식을 보면 H3 + H1을 tanh에 넣어주고 있다. 즉, 3층의 입력은 2층의 출력과 1층의 출력을 더한 값이 된다.

        # 4번째 출력층: 3층의 출력(z3)을 입력받아 Sigmoid로 확률값(0~1) 산출
        h4 = z3 @ W4 + b4
        y_hat = sigmoid(h4)

        # 차원 맞추기: 계산 결과인 y_hat은 (N, 1) 형태의 2차원 열벡터로 나온다. 뒤쪽 코드에서 (N,) 형태의 1차원 벡터를 원하기 때문에 reshape를 통해 차원을 맞춰준다.
        y_hat = y_hat.reshape(-1)
        #################

        assert y_hat.shape == (X.shape[0],), (
            f"y_hat.shape is {y_hat.shape}. Reshape y_hat to {(X.shape[0],)}"
        )

        cache = {
            "h1": h1, "z1": z1,
            "h2": h2, "z2": z2,
            "h3": h3, "z3": z3,
            "h4": h4, "y_hat": y_hat,
        }
        return y_hat, cache

    def back_propagation(self, cache, X, y, L2_norm=0.0):
        """
        Compute gradients for all parameters.

        Args:
            cache: (dict) Values needed to compute gradients
            X: (numpy array) Input data of shape (N, D)
            y: (numpy array) Training labels (N, ) -> (N, 1)
            L2_norm: (int) L2 normalization coefficient
                    
        Returns:
            grads: (dict) Dictionary mapping parameter names to gradients of model parameters
        """
        W1, W2, W3, W4 = (
            self.model["W1"],
            self.model["W2"],
            self.model["W3"],
            self.model["W4"],
        )
        h1, z1 = cache["h1"], cache["z1"]
        h2, z2 = cache["h2"], cache["z2"]
        h3, z3 = cache["h3"], cache["z3"]
        y_hat = cache["y_hat"]

        # For matrix computation
        # 행렬 계산을 위한 Shape 맞추기 (N, ) -> (N, 1)

        y = y.reshape(-1, 1)
        y_hat = y_hat.reshape(-1, 1)

        ############################################################
        # gradient 계산하는 과정

        dy_hat = (y_hat - y) / (y_hat * (1 - y_hat) + 1e-15)

        dh4 = dy_hat * y_hat * (1 - y_hat)
        db4 = np.sum(dh4, axis=0, keepdims=True)
        dW4 = z3.T @ dh4 + 2 * L2_norm * W4
        dz3 = dh4 @ W4.T

        ### CODE HERE ###
        # 'gradient 계산하는 과정'을 참고하여 gradient 계산
        # [Layer 3 역전파]
        # dh3, db3, dW3, dz2
        # # z3 = tanh(h3 + h1) 이므로, tanh의 미분 공식인 (1 - tanh^2)을 적용합니다.
        dtanh = dz3 * (1 - z3 ** 2)
        dh3 = dtanh
        db3 = np.sum(dh3, axis=0, keepdims=True) # 데이터 N개 전체에 대한 오차를 하나로 합쳐야 하므로 행 방향(axis=0)으로 모두 더합니다. keepdims=True를 통해 결과가 (1, 10) 형태의 2차원 행렬로 유지되도록 합니다.
        dW3 = z2.T @ dh3 + 2 * L2_norm * W3 # 가중치 기울기 구하기: (이전 층의 입력 데이터의 전치 행렬) @ (현재 층의 기울기 dh) + L2 정규화 미분항
        dz2 = dh3 @ W3.T # 현재 층의 기울기 dh와 현재 층의 가중치 W의 전치 행렬을 곱하여 이전 층으로 전달할 기울기 dz를 계산합니다.

        # [Layer 2 역전파]
        # dh2, db2, dW2, dz1
        dh2 = dz2 * leakyrelu_grad(h2)  # 전달받은 기울기(dz)에 해당 층 활성화 함수의 미분값을 요소별로 곱합니다.
        db2 = np.sum(dh2, axis=0, keepdims=True)
        dW2 = z1.T @ dh2 + 2 * L2_norm * W2
        dz1 = dh2 @ W2.T

        # [Layer 1 역전파]
        # dh1, db1, dW1
        # h1에 대해 ReLU의 미분 함수(relu_grad)를 곱해줍니다.
        # 순전파 때 z3 = tanh(h3 + h1) 로 Skip Connection이 존재했으므로, z3에서 건너온 dtanh 기울기도 h1에 함께 더해져야(+=) 합니다!
        dh1 = (dz1 * relu_grad(h1)) + dtanh
        db1 = np.sum(dh1, axis=0, keepdims=True)
        dW1 = X.T @ dh1 + 2 * L2_norm * W1
        #################
        ############################################################

        grads = {
            "dy_hat": dy_hat,
            "dh4": dh4,
            "dW4": dW4,
            "db4": db4,
            "dW3": dW3,
            "db3": db3,
            "dW2": dW2,
            "db2": db2,
            "dW1": dW1,
            "db1": db1,
        }
        return grads

    def compute_loss(self, y_pred, y_true, L2_norm=0.0):
        """
        Descriptions: Evaluate the total loss on the dataset
                
        Args:
            y_pred: (numpy array) Predicted target (N,)
            y_true: (numpy array) Array of training labels (N,)
                
        Returns:
            loss: (float) Loss (data loss and regularization loss) for training samples.BCE loss + L2 regularization.
        """
        W1, W2, W3, W4 = (
            self.model["W1"],
            self.model["W2"],
            self.model["W3"],
            self.model["W4"],
        )

        y_true = y_true.reshape(-1, 1)
        y_pred = y_pred.reshape(-1, 1)
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)

        log_loss = -np.sum(y_true * np.log(y_pred)+ (1 - y_true) * np.log(1 - y_pred))
        l2_loss = L2_norm * (np.sum(W1 ** 2) + np.sum(W2 ** 2) + np.sum(W3 ** 2) + np.sum(W4 ** 2))
        total_loss = log_loss + l2_loss

        return total_loss

    def train(
        self,
        X_train,
        y_train,
        X_val=None,
        y_val=None,
        learning_rate=1e-3,
        L2_norm=0.0,
        epoch=20000,
        print_loss=True,
        optimizer="bgd",
        batch_size=32,
        seed=0,
        eval_every=10,
    ):
        """
        Train with BGD, SGD, or Mini-batch GD.

        optimizer:
            'bgd'       -> full batch
            'sgd'       -> batch size 1
            'minibatch' -> user-defined batch_size
        """
        if optimizer not in {"bgd", "sgd", "minibatch"}:
            raise ValueError("optimizer must be 'bgd', 'sgd', or 'minibatch'")

        n = X_train.shape[0]

        if optimizer == "bgd":
            effective_batch_size = n
        elif optimizer == "sgd":
            effective_batch_size = 1
        else:
            if batch_size is None or batch_size <= 0:
                raise ValueError("batch_size must be a positive integer")
            effective_batch_size = min(batch_size, n)

        rng = np.random.default_rng(seed)

        history = {
            "epoch": [],
            "update_step": [],
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
            "elapsed_sec": [],
            # backward-compatible keys
            "loss_history": [],
            "train_acc_history": [],
            "val_acc_history": [],
        }

        update_step = 0
        start_time = time.perf_counter()

        for it in range(epoch):
            indices = rng.permutation(n)

            for start in range(0, n, effective_batch_size):
                idx = indices[start:start + effective_batch_size]
                X_batch = X_train[idx]
                y_batch = y_train[idx]

                ### CODE HERE ###
                #################
                # 1. forward propagation(순전파 수행): 현재 Mini-batch 데이터로 예측값과 캐시 계산
                y_hat, cache = self.forward_propagation(X_batch)

                # 2. loss 계산: 예측값과 실제 정답 사이의 오차 산출
                loss = self.compute_loss(y_hat, y_batch, L2_norm)

                # 3. back propagation 수행 후 gradient update
                grad = self.back_propagation(cache, X_batch, y_batch, L2_norm)

                # 4. 경사하강법(Gradient Descent) 적용: 가중치와 편향 업데이트 (기울기가 양수이면 가중치를 줄이고 음수이면 가중치를 늘려서 손실이 최소가 되는 골짜기로 내려가는 원리)
                # 파라미터 = 기존 파라미터 - (학습률 * 기울기)
                self.model["W1"] -= learning_rate * grad["dW1"]
                self.model["b1"] -= learning_rate * grad["db1"]
                self.model["W2"] -= learning_rate * grad["dW2"]
                self.model["b2"] -= learning_rate * grad["db2"]
                self.model["W3"] -= learning_rate * grad["dW3"]
                self.model["b3"] -= learning_rate * grad["db3"]
                self.model["W4"] -= learning_rate * grad["dW4"]
                self.model["b4"] -= learning_rate * grad["db4"]
                #################
                update_step += 1

            if (it + 1) % eval_every == 0 or (it + 1) == epoch:
                train_prob, _ = self.forward_propagation(X_train)
                train_loss = self.compute_loss(train_prob, y_train, L2_norm)
                train_pred = self.predict(X_train)
                train_acc = np.mean(train_pred == y_train)

                history["epoch"].append(it + 1)
                history["update_step"].append(update_step)
                history["train_loss"].append(float(train_loss))
                history["train_acc"].append(float(train_acc))
                history["loss_history"].append(float(train_loss))
                history["train_acc_history"].append(float(train_acc))
                history["elapsed_sec"].append(time.perf_counter() - start_time)

                if X_val is not None and y_val is not None:
                    val_prob, _ = self.forward_propagation(X_val)
                    val_loss = self.compute_loss(val_prob, y_val, L2_norm=0.0)
                    val_pred = self.predict(X_val)
                    val_acc = np.mean(val_pred == y_val)

                    history["val_loss"].append(float(val_loss))
                    history["val_acc"].append(float(val_acc))
                    history["val_acc_history"].append(float(val_acc))

                if print_loss and (
                    (it + 1) % max(eval_every * 10, 1) == 0
                    or (it + 1) == epoch
                ):
                    msg = (
                        f"[{optimizer}] epoch={it + 1}, "
                        f"train_loss={train_loss:.4f}, "
                        f"train_acc={train_acc:.3f}"
                    )
                    if X_val is not None and y_val is not None:
                        msg += (
                            f", val_loss={val_loss:.4f}, "
                            f"val_acc={val_acc:.3f}"
                        )
                    print(msg)

        return history

    def predict(self, X):
        ### CODE HERE ###
        # Binary classification: 0.5 이상이면 1, 아니면 0
        # 입력 데이터 X에 대해 순전파를 수행하여 확률값을 얻습니다. 그 후, 0.5를 기준으로 이진 분류 결과를 반환합니다.
        y_hat, _ = self.forward_propagation(X) # 시그모이드 함수를 지났기 때문에 0~1 사이의 확률 값을 갖는다.
        predictions = (y_hat >= 0.5).astype(int) # 불리언 배열을 정수형으로 변환하여 0과 1로 이루어진 배열을 반환한다.
        #################
        return predictions


def tanh(x):
    # tanh(쌍곡선탄젠트) 함수: 입력값을 -1과 1 사이의 값으로 변환
    return np.tanh(x)


def relu(x):
    # ReLU 함수: 입력값이 0보다 크면 그대로 출력, 0 이하이면 0을 출력
    # np.maximum(0, x)는 0과 x의 요소별 최대값을 반환함
    return np.maximum(0, x)


def relu_grad(x):
    # ReLU의 미분(기울기): x > 0 일 때는 기울기가 1, 아니면 0
    # (x > 0)은 True/False 배열을 만들고, .astype(float)로 1.0/0.0 숫자로 변환
    return (x > 0).astype(float)


def leakyrelu(x, alpha=0.01):
    # LeakyReLU 함수: ReLU의 단점(0 이하에서 기울기가 사라지는 현상)을 보완
    # x > 0 일 때는 x 그대로, x <= 0 일 때는 alpha(0.01)를 곱한 아주 작은 값 출력
    # np.where(조건, 조건이 참일때 값, 거짓일때 값)
    return np.where(x > 0, x, alpha * x)


def leakyrelu_grad(x, alpha=0.01):
    # LeakyReLU의 미분(기울기):
    # x > 0 일 때는 기울기가 1.0, x <= 0 일 때는 기울기가 alpha(0.01)
    return np.where(x > 0, 1.0, alpha)

def sigmoid(x):
    # Sigmoid(시그모이드) 함수: 입력값을 0과 1 사이의 확률값으로 변환
    # 이진 분류(Binary Classification)의 최종 출력층에서 주로 사용됨
    return 1 / (1 + np.exp(-x))