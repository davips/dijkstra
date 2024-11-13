# Aula: Classificadores e Regressões Populares em Python
# Este script fornece exemplos de implementação dos seis classificadores mais populares e três técnicas de regressão
# utilizando bibliotecas populares do Python, como Scikit-learn. Cada seção de código é comentada para facilitar o
# entendimento dos conceitos e do funcionamento de cada modelo.

from sklearn.datasets import fetch_california_housing
# Importando bibliotecas necessárias
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# Carregar dados para classificação e regressão
# Utilizaremos o dataset 'Iris' para classificação e o dataset 'Boston' para regressão
data_classification = load_breast_cancer()
data_regression = fetch_california_housing()

X_class, y_class = data_classification.data, data_classification.target
X_reg, y_reg = data_regression.data, data_regression.target

# Dividindo os dados em treino e teste
X_class_train, X_class_test, y_class_train, y_class_test = train_test_split(X_class, y_class, test_size=0.3, random_state=42)
X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(X_reg, y_reg, test_size=0.3, random_state=42)

# Normalização dos dados
scaler = StandardScaler()
X_class_train = scaler.fit_transform(X_class_train)
X_class_test = scaler.transform(X_class_test)
X_reg_train = scaler.fit_transform(X_reg_train)
X_reg_test = scaler.transform(X_reg_test)

# --- CLASSIFICADORES ---

# 1. K-Nearest Neighbors (KNN)
# Classificador que considera a classe mais frequente entre os k vizinhos mais próximos.
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_class_train, y_class_train)
y_pred_knn = knn.predict(X_class_test)
acc_knn = accuracy_score(y_class_test, y_pred_knn)

# 2. Decision Tree Classifier
# Árvore de decisão que particiona o espaço de busca em regiões de decisão baseadas em características dos dados.
dtc = DecisionTreeClassifier(random_state=42)
dtc.fit(X_class_train, y_class_train)
y_pred_dtc = dtc.predict(X_class_test)
acc_dtc = accuracy_score(y_class_test, y_pred_dtc)

# 3. Support Vector Classifier (SVC)
# Algoritmo que encontra uma hipersuperfície que melhor separa as classes no espaço de características.
svc = SVC(kernel='linear', random_state=42)
svc.fit(X_class_train, y_class_train)
y_pred_svc = svc.predict(X_class_test)
acc_svc = accuracy_score(y_class_test, y_pred_svc)

# 4. Naive Bayes (GaussianNB)
# Classificador probabilístico baseado no Teorema de Bayes, com suposição de independência entre características.
gnb = GaussianNB()
gnb.fit(X_class_train, y_class_train)
y_pred_gnb = gnb.predict(X_class_test)
acc_gnb = accuracy_score(y_class_test, y_pred_gnb)

# 5. Logistic Regression
# Modelo que aplica uma função logística para estimar a probabilidade de uma amostra pertencer a uma classe.
lr = LogisticRegression(random_state=42, max_iter=200)
lr.fit(X_class_train, y_class_train)
y_pred_lr = lr.predict(X_class_test)
acc_lr = accuracy_score(y_class_test, y_pred_lr)

# 6. Random Forest Classifier
# Ensemble de várias árvores de decisão, que combina suas previsões para melhorar a precisão.
rfc = RandomForestClassifier(n_estimators=100, random_state=42)
rfc.fit(X_class_train, y_class_train)
y_pred_rfc = rfc.predict(X_class_test)
acc_rfc = accuracy_score(y_class_test, y_pred_rfc)

# Exibindo acurácias dos classificadores
print("Acurácias dos classificadores:")
print(f"K-Nearest Neighbors: {acc_knn:.2f}")
print(f"Decision Tree: {acc_dtc:.2f}")
print(f"Support Vector Classifier: {acc_svc:.2f}")
print(f"Naive Bayes: {acc_gnb:.2f}")
print(f"Logistic Regression: {acc_lr:.2f}")
print(f"Random Forest: {acc_rfc:.2f}")

# --- TÉCNICAS DE REGRESSÃO ---

# 1. Regressão Linear
# Método básico de regressão que encontra uma linha de melhor ajuste para os dados.
lin_reg = LinearRegression()
lin_reg.fit(X_reg_train, y_reg_train)
y_pred_lin = lin_reg.predict(X_reg_test)
mse_lin = mean_squared_error(y_reg_test, y_pred_lin)

# 2. Regressão Ridge
# Regressão linear com regularização L2, penalizando os coeficientes grandes para evitar overfitting.
ridge = Ridge(alpha=1.0)
ridge.fit(X_reg_train, y_reg_train)
y_pred_ridge = ridge.predict(X_reg_test)
mse_ridge = mean_squared_error(y_reg_test, y_pred_ridge)

# 3. Regressão Lasso
# Regressão linear com regularização L1, que pode reduzir coeficientes irrelevantes a zero.
lasso = Lasso(alpha=0.1)
lasso.fit(X_reg_train, y_reg_train)
y_pred_lasso = lasso.predict(X_reg_test)
mse_lasso = mean_squared_error(y_reg_test, y_pred_lasso)

# Exibindo erros quadráticos médios das regressões
print("\nErros Quadráticos Médios das regressões:")
print(f"Regressão Linear: {mse_lin:.2f}")
print(f"Regressão Ridge: {mse_ridge:.2f}")
print(f"Regressão Lasso: {mse_lasso:.2f}")
