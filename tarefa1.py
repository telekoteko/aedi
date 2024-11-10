# AEDI - Tarefa 1 - 2024
import streamlit as sl
import numpy as np
import pandas as pd
import random
from scipy.stats import poisson

sl.title("AEDI - Simulação de Jogos do Brasileirão 2024")
sl.write("Uma aplicação que usa a distribuição de Poisson e Simulação de Monte Carlo para estimar resultados de partidas de futebol a partir do histórico (médias) de gols de cada time como mandante e visitante ao longo das 32 rodadas já encerradas no Brasileirão Série A 2024.")

# carregar os dados
df = pd.read_csv("brasileirao_2024.csv")
times = df['nome_mandante'].unique()

# funcao de callback p atualizar o session_state quando o selectbox muda
def atualizar_times():
    sl.session_state['time1'] = sl.session_state['selecionado_time1']
    sl.session_state['time2'] = sl.session_state['selecionado_time2']

# seleção inicial aleatória dos times
if 'time1' not in sl.session_state:
    sl.session_state['time1'], sl.session_state['time2'] = random.sample(list(times), 2)

# pra capturar seleção do usuário
if 'selecionado_time1' not in sl.session_state:
    sl.session_state['selecionado_time1'] = sl.session_state['time1']
if 'selecionado_time2' not in sl.session_state:
    sl.session_state['selecionado_time2'] = sl.session_state['time2']

# selectbox de times
time1 = sl.selectbox(
    "Selecione o time mandante:", times, 
    index=list(times).index(sl.session_state['selecionado_time1']),
    key='selecionado_time1', on_change=atualizar_times
)

time2 = sl.selectbox(
    "Selecione o time visitante:", times, 
    index=list(times).index(sl.session_state['selecionado_time2']),
    key='selecionado_time2', on_change=atualizar_times
)

# filtrar dados por time (como mandante ou visitante)
time1_jogos_mandante = df[df['nome_mandante'] == sl.session_state['time1']]
time2_jogos_visitante = df[df['nome_visitante'] == sl.session_state['time2']]

# calcular a média de gols (como mandante ou visitante)
media_gols_time1_mandante = time1_jogos_mandante['gols_mandante'].mean()
media_gols_time2_visitante = time2_jogos_visitante['gols_visitante'].mean()

sl.write(f"Média de gols do {time1} como mandante: {media_gols_time1_mandante:.2f}")
sl.write(f"Média de gols do {time2} como visitante: {media_gols_time2_visitante:.2f}")

# Poisson é útil para modelar o número de gols esperados em partidas de futebol, 
# onde eventos (gols) acontecem de forma independente e em intervalos aleatórios.

# a função de simulação de partidas
def simular_partida(time1_media_gols, time2_media_gols, num_simulacoes=10000):
    gols_time1 = poisson.rvs(mu=time1_media_gols, size=num_simulacoes)
    gols_time2 = poisson.rvs(mu=time2_media_gols, size=num_simulacoes)
    resultado = {
        "vitorias_mandante": np.sum(gols_time1 > gols_time2),
        "empates": np.sum(gols_time1 == gols_time2),
        "vitorias_visitante": np.sum(gols_time1 < gols_time2)
    }
    return resultado, gols_time1, gols_time2

# slider p ajustar o número de simulações
num_simulacoes = sl.slider("Número de simulações", min_value=100, max_value=50000, value=10000, step=1000)

# sliders para ajustar média de gols do mandante e visitante para ver impactos nas previsões
media_gols_time1_mandante_customizada = sl.slider(f"Média de gols para {time1} (como mandante)", 0.0, 5.0, media_gols_time1_mandante, 0.1)
media_gols_time2_visitante_customizada = sl.slider(f"Média de gols para {time2} (como visitante)", 0.0, 5.0, media_gols_time2_visitante, 0.1)

# simular a partida com os parâmetros ajustados
resultado, gols_time1, gols_time2 = simular_partida(media_gols_time1_mandante_customizada, media_gols_time2_visitante_customizada, num_simulacoes)

# exibir os resultados
sl.subheader("Resultado da Simulação")
sl.write(f"Após {num_simulacoes} simulações:")
sl.write(f"Vitórias do {time1}: {resultado['vitorias_mandante']} ({resultado['vitorias_mandante'] / num_simulacoes:.2%})")
sl.write(f"Empates: {resultado['empates']} ({resultado['empates'] / num_simulacoes:.2%})")
sl.write(f"Vitórias do {time2}: {resultado['vitorias_visitante']} ({resultado['vitorias_visitante'] / num_simulacoes:.2%})")

# histograma
sl.subheader("Histograma da distribuição dos gols simulados")

gols_time1_distribuicao = pd.Series(gols_time1).value_counts().sort_index()
gols_time2_distribuicao = pd.Series(gols_time2).value_counts().sort_index()

# montar o df da distribuicao
distribuicao_gols_df = pd.DataFrame({
    f"Gols {time1} (Mandante)": gols_time1_distribuicao,
    f"Gols {time2} (Visitante)": gols_time2_distribuicao
}).fillna(0)

# plotar o gráfico de distribuição de gols
sl.bar_chart(distribuicao_gols_df)

# pq usei poisson?
sl.subheader("Como as probabilidades são calculadas?")
sl.write("""
        A simulação proposta tem como base a previsão de gols numa partida futura, considerando o histórico de cada time como mandante e visitante no Brasileirão 2024 pra gerar dados estatísticos que a apoiem.
        \n Pra isso, foi aplicado o conceito da Simulação de Monte Carlo, que envolve repetir várias vezes um processo de sorteio de eventos aleatórios com base numa distribuição de probabilidade escolhida de acordo com o problema.
        \n Nesse caso, os eventos aleatórios são o número de gols. E a distribuição que se adequou ao caso foi a de Poisson.
        Com diversas simulações, é possível olhar a distribuição de resultados e calcular a probabilidade de vitória, empate ou derrota.""")
sl.subheader("Por que a distribuição de Poisson?")
sl.write("""
        A distribuição de Poisson foi escolhida porque modela a contagem de eventos independentes que ocorrem numa taxa média constante ao longo de um intervalo fixo.
         \n Nesse caso, os 90 minutos de uma partida são um intervalo fixo.
         \n Os gols ocorrem de forma independente, de certa forma, pois um gol não depende diretamente de outro ter ocorrido.
        \n Gols também são eventos inteiros e discretos, o que se adequa à Poisson.
         \n A taxa constante de ocorrÊncia (λ) foi possível ser estimada devido cada time ter uma média de gols por partida, que se mantém relativamente estável quando se observa várias partidas como mandante ou visitante.
         \n Na distribuição de Poisson, a taxa λ, usada para definir a média de gols esperada, determina a variabilidade de gols simulados.  Isso faz sentido nos jogos de futebol, pois times com médias de gols mais altas têm uma distribuição simulada que permite mais resultados possíveis, enquanto times com médias baixas terão uma maior concentração em resultados próximos de zero.
         \n Nessa distribuição, a probabilidade de ocorrer um número maior de gols (os eventos) diminui de forma exponencial conforme x cresce, o que reflete a realidade do futebol, em que é mto difícil um time fazer 5 gols, por exemplo. [Fórmula: P(X = x) = (e^-λ * λ^x) / x!]
        Exemplo: Se um tive tiver média de 1,5 gols por partida (λ=1.5), a chance de fazer 5 gols fica próxima de apenas 4%.
        \n Cabe a reflexão sobre diversas variáveis do futebol não capturada nessa modelagem simples. Expulsoes, estratégias de jogo, etc.
         \n Seria interessante analisar um conjunto maior de jogos, mas uma equipe varia muito de um ano para o outro, então teria que analisar se traria resultados melhores ou piores nas previsões dos próximos jogos da temporada atual temporada. 
          """)

sl.subheader("Sobre os dados")

sl.write("""Fonte: https://olympics.com/pt/noticias/campeonato-brasileiro-2024-todos-resultados
         \n Extraído em 08/11/2024, com 32 rodadas já encerradas no campeonato brasileiro.
         \n Os dados foram copiados, tratados e gravados de forma estruturada num csv para uso na tarefa.
         \n Formato:""")
sl.write(df.head())
