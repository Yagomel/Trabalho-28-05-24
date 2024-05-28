import numpy as np
import pandas as pd 
from sklearn.linear_model import LinearRegression 
import matplotlib.pyplot as plt 

# Gerando dados sintéticos
np.random.seed(42)
temperatura = np.random.rand(100, 1) * 40  # Temperatura em Celsius
hora_do_dia = np.random.rand(100, 1) * 24  # Hora do dia em horas
demanda_energia = (temperatura * 2) + (hora_do_dia * 3) + np.random.randn(100, 1) * 10

# Criando DataFrame
dados = pd.DataFrame(np.concatenate([temperatura, hora_do_dia, demanda_energia], axis=1), 
                     columns=['Temperatura', 'Hora do Dia', 'Demanda de Energia'])

# Preparando os dados para treinamento
X = dados[['Temperatura', 'Hora do Dia']]
y = dados['Demanda de Energia']

# Treinamento do modelo
modelo = LinearRegression().fit(X, y)

# Avaliação do modelo (métricas não foram calculadas)
# Visualização dos resultados
plt.scatter(y, modelo.predict(X))
plt.xlabel('Demanda Real de Energia')
plt.ylabel('Demanda Prevista de Energia')
plt.title('Previsões de Demanda de Energia')
plt.show()
